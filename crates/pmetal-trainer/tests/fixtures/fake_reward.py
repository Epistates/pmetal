#!/usr/bin/env python3
"""A stand-in for a user's reward command, for the trainer's tests.

Speaks the protocol of `ExternalCommandReward`: one JSON request per line on
stdin, one JSON answer per line on stdout. The first argument picks how it
behaves; the others are its parameters.

    length                 reward = number of characters in the completion
    column NAME            reward = float(row[NAME]) of the completion's row
    echo-prompts           reward = float(len(prompts)) for every completion
    keys                   reward = number of request keys (so a test sees which were sent)
    constant X             reward = X
    crash-after N          answer N requests, then exit with status 3 (writes to stderr first)
    hang                   read the request and never answer
    wrong-length           answer with one reward too many
    null                   reward null for the first completion
    nan                    reward "NaN" (a string) for the first completion
    error MESSAGE          answer {"error": MESSAGE}
    garbage                answer with a line that is not JSON
    spawn-child PIDFILE    like length, after starting a sleeping child whose pid goes in PIDFILE
"""

import json
import subprocess
import sys
import time


def main() -> None:
    mode, *params = sys.argv[1:]
    answered = 0

    if mode == "spawn-child":
        child = subprocess.Popen(["sleep", "300"])
        with open(params[0], "w") as f:
            f.write(str(child.pid))

    for line in sys.stdin:
        request = json.loads(line)
        completions = request["completions"]
        n = len(completions)

        if mode in ("length", "spawn-child"):
            rewards = [float(len(c)) for c in completions]
        elif mode == "column":
            rewards = [float(v) for v in request[params[0]]]
        elif mode == "echo-prompts":
            rewards = [float(len(request["prompts"]))] * n
        elif mode == "keys":
            rewards = [float(len(request))] * n
        elif mode == "constant":
            rewards = [float(params[0])] * n
        elif mode == "crash-after":
            if answered >= int(params[0]):
                print("scoring service lost its database", file=sys.stderr, flush=True)
                sys.exit(3)
            rewards = [1.0] * n
        elif mode == "hang":
            time.sleep(300)
            return
        elif mode == "wrong-length":
            rewards = [1.0] * (n + 1)
        elif mode == "null":
            rewards = [None] + [1.0] * (n - 1)
        elif mode == "nan":
            rewards = ["NaN"] + [1.0] * (n - 1)
        elif mode == "error":
            print(json.dumps({"error": params[0]}), flush=True)
            answered += 1
            continue
        elif mode == "garbage":
            print("this is not json", flush=True)
            continue
        else:
            raise SystemExit(f"unknown mode {mode}")

        print(json.dumps({"rewards": rewards}), flush=True)
        answered += 1


if __name__ == "__main__":
    main()
