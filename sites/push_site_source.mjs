import { spawnSync } from "node:child_process";
const stdin = process.stdin;
const value = await new Promise((resolve, reject) => {
  let data = "";
  const finish = (error) => {
    stdin.off("data", onData);
    stdin.off("end", onEnd);
    if (stdin.isTTY) stdin.setRawMode(false);
    stdin.pause();
    error ? reject(error) : resolve(data);
  };
  const onData = (chunk) => {
    data += chunk;
    if (data.includes("\u0003") || data.length > 65536) finish(new Error("Site credential input cancelled."));
    else if (/[\r\n]/.test(data)) finish();
  };
  const onEnd = () => finish();
  if (stdin.isTTY) stdin.setRawMode(true);
  console.log("Ready for Site credential on stdin (input is hidden).");
  stdin.setEncoding("utf8");
  stdin.on("data", onData);
  stdin.once("end", onEnd);
  stdin.resume();
});
let credential;
try {
  credential = JSON.parse(value);
} catch {
  throw new Error("Could not parse the Site credential.");
}
const remote = new URL(credential.remote_url);
if (remote.origin !== "https://git.chatgpt-team.site" || remote.username || remote.password ||
    credential.branch !== "main" || credential.auth_mode !== "http_extra_header" ||
    !credential.token || /[\r\n\0]/.test(credential.token)) {
  throw new Error("Unexpected Site repository credential.");
}
const sha = process.argv[2];
if (!/^[0-9a-f]{40}$/.test(sha)) throw new Error("Expected a full local commit ID.");
const local = spawnSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" });
if (local.status !== 0 || local.stdout.trim() !== sha) throw new Error("Local HEAD changed.");
const env = { ...process.env, OSA_SITES_AUTHORIZATION: `Authorization: Bearer ${credential.token}`,
              GIT_TERMINAL_PROMPT: "0" };
const args = ["-c", "credential.helper=", "-c", "http.extraHeader=", "-c", "http.followRedirects=false",
              `--config-env=http.${remote.href}.extraHeader=OSA_SITES_AUTHORIZATION`];
const ref = "refs/heads/main";
const push = spawnSync("git", [...args, "push", remote.href, `${sha}:${ref}`], { env, encoding: "utf8" });
if (push.status !== 0) {
  console.error((push.stderr || "Git push did not complete.").split(credential.token).join("[redacted]"));
  process.exitCode = 1;
} else {
  const status = spawnSync("git", [...args, "ls-remote", "--heads", remote.href, ref], { env, encoding: "utf8" });
  if (status.status !== 0 || status.stdout.trim().split(/\s+/)[0] !== sha) {
    throw new Error("The source commit did not match the remote branch.");
  }
  console.log(`Verified remote commit ${sha}`);
}
