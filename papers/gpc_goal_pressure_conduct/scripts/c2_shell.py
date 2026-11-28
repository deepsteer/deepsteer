#!/usr/bin/env python3
"""C2's shell emulator (GPC_PLAN.md §19.16; the lead's 2026-11-27 3:05 PM item 2), rebuilt from the room pilot's
command logs: 96 loops, 1,200 commands, read by script. The pilot's emulator mapped `/app` to nothing, printed names
for `ls -la`, and lacked `cd`, `find`, `which`, `awk` and a dozen more, so 93 of 96 loops ran out of steps exploring.

**The working directory** is `/app`, the tree's root; `cd` moves within it. A path is resolved against it: `/app/x`,
`./x`, `x`, `..`, and `$PWD`/`~` expansions. A path outside `/app` resolves to nothing (No such file or directory),
as on a machine where the task's files are only under `/app`.

**The shell:** chains (`&&`, `||`, `;`, newlines) and pipes outside quotes; redirects (`> f`, `>> f`, `2>&1`,
`2>/dev/null`, `>/dev/null`, `&>`); globs (`*`, `?`, `[...]`) against the tree; `$(...)` and backticks; `for V in
LIST; do BODY; done`; heredocs to a file.

**The commands, with the arguments the pilot used:** `ls` (`-l -a -R -1 -d -h -t -S`, several paths, files),
`cat` (`-n -A`), `head`/`tail` (`-n N`, `-N`, `-c N`), `grep` (`-i -c -n -v -l -r -E -F -w`), `wc` (`-l -c -w`,
several files), `pwd`, `cd`, `echo` (`-n -e`), `printf`, `sed` (`-i s///`, `-n Np`), `cp`, `mv`, `rm`, `touch`,
`mkdir`, `find` (`-maxdepth -type -name -iname -exec cmd {} ; -print`), `which`, `type`, `command -v`, `stat`,
`file`, `od -c`, `xxd`, `date`, `env`, `printenv`, `whoami`, `id`, `readlink -f`, `realpath`, `basename`,
`dirname`, `true`, `false`, `chmod`, `export`, `alias`, `uname`, `md5sum`, `hexdump`, `sort` (`-n -r -u -k`), `uniq` (`-c`), `tr`, `cut` (`-d -f -c`), `awk` (a
subset: `-F`, `BEGIN`/`END`, conditions on `NR`, `NF` and fields, `+=`, `=`, `print`, `printf`), `bash -c` and
`sh -c`, and `/bin/<name>`, `/usr/bin/<name>`.

**Left unanswered, and why:** `python3 -c` and `python -c` (arbitrary code; the real-bash sandbox, §19.16, is the
alternative that answers them); `if … then … fi` and other compound syntax beyond `for` (one use in the pilot); and
any command not above (answered as a shell answers a missing tool, exit 127).
"""

from __future__ import annotations

import fnmatch
import re
import shlex

UNCOVERED_EXIT = 127
APP = "/app"
DATE = "Fri Nov 27 15:00:00 HST 2026"
ENV = {"PWD": APP, "HOME": "/home/agent", "USER": "agent", "SHELL": "/bin/bash", "PATH": "/usr/local/bin:/usr/bin:/bin",
       "LANG": "C.UTF-8"}
STAMP = "Nov 27 09:00"


def split_top(s: str, seps: tuple[str, ...]) -> list[tuple[str, str]]:
    """Split on separators outside quotes and $( ): returns (piece, the separator that followed it)."""
    out, buf, q, depth, i = [], [], None, 0, 0
    while i < len(s):
        ch = s[i]
        if q:
            buf.append(ch)
            if ch == "\\" and q == '"' and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if ch == q:
                q = None
            i += 1
            continue
        if ch in "'\"":
            q = ch
            buf.append(ch)
            i += 1
            continue
        if s.startswith("$(", i):
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
        if depth == 0:
            hit = next((sp for sp in seps if s.startswith(sp, i)), None)
            if hit:
                out.append(("".join(buf), hit))
                buf = []
                i += len(hit)
                continue
        buf.append(ch)
        i += 1
    out.append(("".join(buf), ""))
    return out


class Shell:
    """The shell over `self.tree` (path -> text, relative to /app) and `self.dirs` (explicit directories)."""

    tree: dict[str, str]

    def _shell_init(self) -> None:
        self.cwd = ""
        self.dirs: set[str] = set()

    # ---- paths ----------------------------------------------------------------------------------------------------
    def _abs(self, path: str) -> str | None:
        """A path as the tree's key ('' is /app), or None when it lies outside /app."""
        path = path.replace("$PWD", APP + ("/" + self.cwd if self.cwd else "")).replace("${PWD}", APP)
        if path.startswith("~") or path.startswith("$HOME"):
            return None
        if path == APP or path.startswith(APP + "/"):
            parts = path[len(APP):].split("/")
        elif path.startswith("/"):
            return None
        else:
            parts = (self.cwd.split("/") if self.cwd else []) + path.split("/")
        out: list[str] = []
        for p in parts:
            if p in ("", "."):
                continue
            if p == "..":
                if not out:
                    return None
                out.pop()
            else:
                out.append(p)
        return "/".join(out)

    def _p(self, path: str) -> str:
        """The tree key for a path (kept for the scorer and the outcome code); outside /app, a key no file has."""
        a = self._abs(path)
        return a if a is not None else "\0outside"

    def _is_dir(self, key: str | None) -> bool:
        if key is None:
            return False
        if key == "":
            return True
        return key in self.dirs or any(p.startswith(key + "/") for p in self.tree)

    def _children(self, key: str) -> list[str]:
        pre = key + "/" if key else ""
        names = {p[len(pre):].split("/")[0] for p in self.tree if p.startswith(pre)}
        names |= {d[len(pre):].split("/")[0] for d in self.dirs if d.startswith(pre) and d != key}
        return sorted(n for n in names if n)

    def _glob(self, arg: str) -> list[str]:
        if not re.search(r"[*?\[]", arg):
            return [arg]
        prefix = arg if arg.startswith("/") else None
        pat_key = self._abs(arg.replace("*", "\x01").replace("?", "\x02"))
        if pat_key is None:
            return [arg]
        pat = pat_key.replace("\x01", "*").replace("\x02", "?")
        cands = set(self.tree) | self.dirs | {"/".join(p.split("/")[:k]) for p in self.tree for k in range(1, p.count("/") + 1)}
        hits = sorted(c for c in cands if fnmatch.fnmatchcase(c, pat) and c.count("/") == pat.count("/"))
        if not hits:
            return [arg]
        base = self.cwd + "/" if self.cwd else ""
        return [APP + "/" + h if prefix else (h[len(base):] if h.startswith(base) else APP + "/" + h) for h in hits]

    # ---- the shell --------------------------------------------------------------------------------------------------
    def _chain(self, command: str) -> tuple[str, int]:
        hd = re.match(r"^\s*cat\s*(?:>\s*(\S+)\s*<<-?\s*['\"]?(\w+)['\"]?|<<-?\s*['\"]?(\w+)['\"]?\s*>\s*(\S+))\n(.*)\n\s*"
                      r"(\w+)\s*$", command, re.S)
        if hd and (hd.group(2) or hd.group(3)) == hd.group(6):
            key = self._abs(hd.group(1) or hd.group(4))
            if key is None:
                return f"bash: {hd.group(1) or hd.group(4)}: No such file or directory\n", 1
            self.tree[key] = hd.group(5) + "\n"
            return "", 0
        loop = re.match(r"^\s*for\s+(\w+)\s+in\s+(.*?)\s*;\s*do\s+(.*?)\s*;?\s*done\s*$", command, re.S)
        if loop:
            var, items, body = loop.groups()
            outs, code = [], 0
            for it in [g for w in shlex.split(items) for g in self._glob(w)]:
                o, code = self._chain(body.replace("${" + var + "}", it).replace("$" + var, it))
                outs.append(o)
            return "".join(outs), code
        outs, code, skip = [], 0, False
        for piece, sep in split_top(command.replace("\\\n", " "), ("&&", "||", ";", "\n")):
            if not piece.strip():
                continue
            if not skip:
                o, code = self._pipeline(piece.strip())
                outs.append(o)
            skip = (sep == "&&" and code != 0) or (sep == "||" and code == 0)
        return "".join(outs), code

    def _pipeline(self, piece: str) -> tuple[str, int]:
        data, code = None, 0
        for st, _ in split_top(piece, ("|",)):
            if st.strip():
                data, code = self._one(st.strip(), data)
        return data or "", code

    def _subst(self, cmd: str) -> str:
        def run(m):
            return self._chain(m.group(1))[0].strip()
        cmd = re.sub(r"\$\(([^()]*)\)", run, cmd)
        return re.sub(r"`([^`]*)`", run, cmd)

    def _one(self, cmd: str, stdin: str | None) -> tuple[str, int]:
        cmd = self._subst(cmd)
        # redirects
        drop_err = bool(re.search(r"\s2>\s*/dev/null|&>\s*/dev/null", cmd))
        drop_out = bool(re.search(r"(?<![\d&])>\s*/dev/null|&>\s*/dev/null", cmd))
        cmd = re.sub(r"\s*(?:2>&1|1>&2|>&2|&>\s*/dev/null|2>\s*/dev/null|(?<![\d&])>\s*/dev/null)", "", cmd)
        to = re.search(r"\s*(>>?)\s*([^\s|&;]+)\s*$", cmd)
        if to and not re.search(r"<<", cmd):
            cmd = cmd[: to.start()]
        try:
            argv = shlex.split(cmd)
        except ValueError:
            return f"bash: syntax error near `{cmd[:40]}`\n", 2
        if not argv:
            return stdin or "", 0
        argv = [x for a in argv for x in self._glob(a)]
        argv = [ENV.get(a[1:], a) if re.fullmatch(r"\$[A-Z_]+", a) else a for a in argv]
        name, args = argv[0], argv[1:]
        if re.fullmatch(r"\w+=\S*", name):  # an assignment: nothing to run
            return "", 0
        name = re.sub(r"^(/usr)?(/local)?/bin/", "", name)
        out, code = self._dispatch(name, args, stdin)
        if to:
            key = self._abs(to.group(2))
            if key is None:
                return f"bash: {to.group(2)}: No such file or directory\n", 1
            self.tree[key] = (self.tree.get(key, "") if to.group(1) == ">>" else "") + out
            out = ""
        if drop_out:
            out = ""
        if drop_err and code != 0:
            out = ""
        return out, code

    def _dispatch(self, name: str, args: list[str], stdin: str | None) -> tuple[str, int]:
        scorer = self._scorer_of(name, args)
        if scorer is not None:
            return self.family.scorers[scorer](self.tree)
        if name in ("bash", "sh") and args[:1] == ["-c"] and len(args) > 1:
            return self._chain(args[1])
        if name in ("python3", "python") and args[:1] == ["-c"]:
            return f"bash: {name} -c: command not found\n", UNCOVERED_EXIT
        fn = getattr(self, f"_c_{name.replace('-', '_')}", None) if re.fullmatch(r"[\w-]+", name) else None
        if fn is None:
            return f"bash: {name}: command not found\n", UNCOVERED_EXIT
        return fn(args, stdin)

    def _scorer_of(self, name: str, args: list[str]) -> str | None:
        if name in ("bash", "sh", "python3", "python") and args and not args[0].startswith("-"):
            cand = self._abs(args[0])
        elif name.startswith(("./", "/", "..")) or "/" in name:
            cand = self._abs(name)
        else:
            return None
        return cand if cand in self.family.scorers and cand in self.tree else None

    # ---- files ------------------------------------------------------------------------------------------------------
    def _read(self, path: str) -> tuple[str | None, str]:
        k = self._p(path)
        return self.tree.get(k), k

    def _flags(self, args: list[str], with_value: tuple[str, ...] = ()) -> tuple[set[str], dict[str, str], list[str]]:
        fl, val, rest, it = set(), {}, [], iter(args)
        for a in it:
            if a in with_value:
                val[a] = next(it, "")
            elif a.startswith("--"):
                fl.add(a)
            elif a.startswith("-") and len(a) > 1 and not re.fullmatch(r"-\d+", a):
                fl |= set(a[1:])
            else:
                rest.append(a)
        return fl, val, rest

    def _long(self, name: str, key: str) -> str:
        if self._is_dir(key):
            return f"drwxr-xr-x 2 agent agent {4096:>6} {STAMP} {name}"
        mode = "-rwxr-xr-x" if key.startswith("bin/") or key.endswith(".sh") else "-rw-r--r--"
        return f"{mode} 1 agent agent {len(self.tree.get(key, '').encode()):>6} {STAMP} {name}"

    def _c_ls(self, args, stdin):
        fl, _, paths = self._flags(args)
        paths = paths or ["."]
        outs, code = [], 0
        for i, p in enumerate(paths):
            key = self._abs(p)
            if key is None or (key not in self.tree and not self._is_dir(key)):
                outs.append(f"ls: cannot access '{p}': No such file or directory\n")
                code = 2
                continue
            if key in self.tree or "d" in fl:
                outs.append((self._long(p, key) if "l" in fl else p) + "\n")
                continue
            if "R" in fl:
                stack = [key]
                while stack:
                    k = stack.pop(0)
                    outs.append(f"{APP + ('/' + k if k else '')}:\n" + self._listing(k, fl) + "\n")
                    stack += [(k + "/" if k else "") + c for c in self._children(k) if self._is_dir((k + "/" if k else "") + c)]
                continue
            head = f"{p}:\n" if len(paths) > 1 else ""
            outs.append(head + self._listing(key, fl) + ("\n" if len(paths) > 1 and i < len(paths) - 1 else ""))
        return "".join(outs), code

    def _listing(self, key: str, fl: set[str]) -> str:
        names = self._children(key)
        if "a" in fl or "A" in fl:
            names = ([".", ".."] if "a" in fl else []) + names
        sub = lambda n: n if n in (".", "..") else ((key + "/" if key else "") + n)  # noqa: E731
        if "S" in fl:
            names.sort(key=lambda n: -len(self.tree.get(sub(n), "")))
        if "l" not in fl:
            return "".join(n + "\n" for n in names)
        total = sum(len(self.tree.get(sub(n), "")) for n in names) // 1024 + 4
        return f"total {total}\n" + "".join(self._long(n, key if n == "." else sub(n)) + "\n" for n in names)

    def _c_cat(self, args, stdin):
        fl, _, paths = self._flags(args)
        out, code = [], 0
        for a in paths:
            key = self._abs(a)
            if key is not None and self._is_dir(key) and key not in self.tree:
                out.append(f"cat: {a}: Is a directory\n")
                code = 1
                continue
            body = self.tree.get(key) if key is not None else None
            if body is None:
                out.append(f"cat: {a}: No such file or directory\n")
                code = 1
                continue
            out.append(body)
        text = "".join(out) if paths else (stdin or "")
        if "A" in fl or "e" in fl:
            text = "".join(ln + "$\n" for ln in text.split("\n")[:-1]) if text.endswith("\n") else text
        if "n" in fl:
            text = "".join(f"{i:>6}\t{ln}\n" for i, ln in enumerate(text.splitlines(), 1))
        return text, code

    def _text_of(self, rest: list[str], stdin: str | None, cmd: str) -> tuple[str | None, str | None]:
        if not rest:
            return stdin or "", None
        body = self._read(rest[0])[0]
        return (body, None) if body is not None else (None, f"{cmd}: {rest[0]}: No such file or directory\n")

    def _headtail(self, args, stdin, cmd):
        n, c, rest, it = 10, None, [], iter(args)
        for a in it:
            if a == "-n":
                n = int(next(it, "10").lstrip("+") or 10)
            elif a.startswith("-n") and a[2:].lstrip("+").isdigit():
                n = int(a[2:].lstrip("+"))
            elif a == "-c":
                c = int(next(it, "0"))
            elif re.fullmatch(r"-\d+", a):
                n = int(a[1:])
            elif not a.startswith("-"):
                rest.append(a)
        outs = []
        for p in (rest or [None]):
            text, err = self._text_of([p] if p else [], stdin, cmd)
            if err:
                return err, 1
            if c is not None:
                outs.append(text[:c] if cmd == "head" else text[-c:])
            else:
                lines = text.splitlines()
                pick = lines[:n] if cmd == "head" else lines[-n:] if n else []
                outs.append((f"==> {p} <==\n" if len(rest) > 1 else "") + "".join(x + "\n" for x in pick))
        return "".join(outs), 0

    def _c_head(self, args, stdin):
        return self._headtail(args, stdin, "head")

    def _c_tail(self, args, stdin):
        return self._headtail(args, stdin, "tail")

    def _c_grep(self, args, stdin):
        fl, val, rest = self._flags(args, ("-e",))
        pat = val.get("-e") or (rest.pop(0) if rest else None)
        if pat is None:
            return "grep: no pattern\n", 2
        try:
            rx = re.compile(re.escape(pat) if "F" in fl else (rf"\b{pat}\b" if "w" in fl else pat), re.I if "i" in fl else 0)
        except re.error:
            rx = re.compile(re.escape(pat), re.I if "i" in fl else 0)
        files = rest
        if ("r" in fl or "R" in fl) and not files:
            files = ["."]
        if "r" in fl or "R" in fl:
            keys = []
            for f in files:
                k = self._abs(f)
                keys += [p for p in sorted(self.tree) if k is not None and (p == k or p.startswith(k + "/") or k == "")]
            texts = [(p, self.tree[p]) for p in keys]
        else:
            texts = [(p, self._read(p)[0]) for p in files] if files else [("", stdin or "")]
        many = len(texts) > 1 or "r" in fl or "R" in fl
        out, hits_all = [], 0
        for p, t in texts:
            if t is None:
                out.append(f"grep: {p}: No such file or directory\n")
                continue
            hits = [(i, ln) for i, ln in enumerate(t.splitlines(), 1) if bool(rx.search(ln)) != ("v" in fl)]
            hits_all += len(hits)
            if "l" in fl:
                out += [f"{p}\n"] if hits else []
            elif "c" in fl:
                out.append((f"{p}:" if many else "") + f"{len(hits)}\n")
            else:
                out += [(f"{p}:" if many else "") + (f"{i}:" if "n" in fl else "") + ln + "\n" for i, ln in hits]
        return "".join(out), 0 if hits_all else 1

    def _c_wc(self, args, stdin):
        fl, _, rest = self._flags(args)
        fl = fl or {"l", "w", "c"}

        def counts(t):
            return [n for k, n in (("l", t.count("\n")), ("w", len(t.split())), ("c", len(t.encode()))) if k in fl]
        if not rest:
            return " ".join(f"{n:>7}" for n in counts(stdin or "")) + "\n", 0
        out, tot, code = [], None, 0
        for p in rest:
            t = self._read(p)[0]
            if t is None:
                out.append(f"wc: {p}: No such file or directory\n")
                code = 1
                continue
            cs = counts(t)
            tot = cs if tot is None else [a + b for a, b in zip(tot, cs)]
            out.append(" ".join(f"{n:>7}" for n in cs) + f" {p}\n")
        if len(rest) > 1 and tot:
            out.append(" ".join(f"{n:>7}" for n in tot) + " total\n")
        return "".join(out), code

    def _c_pwd(self, args, stdin):
        return APP + ("/" + self.cwd if self.cwd else "") + "\n", 0

    def _c_cd(self, args, stdin):
        target = args[0] if args else APP
        key = self._abs(target)
        if key is None or not self._is_dir(key):
            return f"bash: cd: {target}: No such file or directory\n", 1
        self.cwd = key
        return "", 0

    def _c_echo(self, args, stdin):
        n = "-n" in args[:1]
        words = [a for a in args if a not in ("-n", "-e", "-ne", "-en")]
        text = " ".join(words)
        if "-e" in args or "-ne" in args or "-en" in args:
            text = text.replace("\\n", "\n").replace("\\t", "\t")
        return text + ("" if n else "\n"), 0

    def _c_printf(self, args, stdin):
        if not args:
            return "", 0
        fmt, vals = args[0].replace("\\n", "\n").replace("\\t", "\t"), list(args[1:])
        try:
            return re.sub(r"%(-?\d*)(s|d)", lambda m: f"{{:{'<' if m.group(1).startswith('-') else '>'}"
                          f"{m.group(1).lstrip('-')}}}".format(vals.pop(0) if vals else ""), fmt).replace("%%", "%"), 0
        except (IndexError, ValueError):
            return fmt, 0

    def _c_sed(self, args, stdin):
        fl, _, rest = self._flags(args)
        if "n" in fl and rest:
            m = re.fullmatch(r"(\d+)(?:,(\d+))?p", rest[0])
            text, err = self._text_of(rest[1:], stdin, "sed")
            if err or not m:
                return err or "sed: unsupported script\n", 1
            a, b = int(m.group(1)), int(m.group(2) or m.group(1))
            return "".join(ln + "\n" for ln in text.splitlines()[a - 1:b]), 0
        if not rest:
            return "sed: no script\n", 1
        m = re.fullmatch(r"s(.)(.*?)\1(.*?)\1(g?)", rest[0])
        if not m:
            return f"sed: -e expression #1: unknown command: `{rest[0][:1]}'\n", 1
        if "i" in fl:
            body, k = self._read(rest[1]) if len(rest) > 1 else (None, "")
            if body is None:
                return f"sed: can't read {rest[1] if len(rest) > 1 else ''}: No such file or directory\n", 2
            self.tree[k] = re.sub(m.group(2), m.group(3), body, count=0 if m.group(4) else 1) if m.group(4) else \
                "".join(re.sub(m.group(2), m.group(3), ln, count=1) + "\n" for ln in body.splitlines())
            return "", 0
        text, err = self._text_of(rest[1:], stdin, "sed")
        if err:
            return err, 2
        return "".join(re.sub(m.group(2), m.group(3), ln, count=0 if m.group(4) else 1) + "\n"
                       for ln in text.splitlines()), 0

    def _c_cp(self, args, stdin):
        _, _, rest = self._flags(args)
        if len(rest) < 2:
            return "cp: missing destination file operand\n", 1
        body, _ = self._read(rest[0])
        if body is None:
            return f"cp: cannot stat '{rest[0]}': No such file or directory\n", 1
        dst = self._abs(rest[1])
        if dst is None:
            return f"cp: cannot create regular file '{rest[1]}': No such file or directory\n", 1
        if self._is_dir(dst) and dst not in self.tree:
            dst = (dst + "/" if dst else "") + rest[0].rstrip("/").split("/")[-1]
        self.tree[dst] = body
        return "", 0

    def _c_mv(self, args, stdin):
        out, code = self._c_cp(args, stdin)
        if code == 0:
            self.tree.pop(self._p([a for a in args if not a.startswith("-")][0]), None)
        return out, code

    def _c_rm(self, args, stdin):
        fl, _, rest = self._flags(args)
        for a in rest:
            k = self._abs(a)
            if k is None:
                continue
            if k in self.tree:
                del self.tree[k]
            elif "r" in fl or "R" in fl:
                for p in [p for p in self.tree if p.startswith(k + "/")]:
                    del self.tree[p]
            elif "f" not in fl:
                return f"rm: cannot remove '{a}': No such file or directory\n", 1
        return "", 0

    def _c_touch(self, args, stdin):
        for a in args:
            k = self._abs(a)
            if k is not None and not a.startswith("-"):
                self.tree.setdefault(k, "")
        return "", 0

    def _c_mkdir(self, args, stdin):
        for a in (x for x in args if not x.startswith("-")):
            k = self._abs(a)
            if k:
                self.dirs |= {"/".join(k.split("/")[:i]) for i in range(1, k.count("/") + 2)}
        return "", 0

    def _c_find(self, args, stdin):
        roots, i, depth, typ, names, exe = [], 0, None, None, [], None
        while i < len(args) and not args[i].startswith("-"):
            roots.append(args[i])
            i += 1
        while i < len(args):
            a = args[i]
            if a == "-maxdepth":
                depth, i = int(args[i + 1]), i + 2
            elif a == "-type":
                typ, i = args[i + 1], i + 2
            elif a in ("-name", "-iname"):
                names.append((args[i + 1], a == "-iname"))
                i += 2
            elif a == "-exec":
                j = next((k for k in range(i + 1, len(args)) if args[k] in (";", "\\;", "+")), len(args))
                exe, i = args[i + 1:j], j + 1
            else:
                i += 1
        out = []
        for r in roots or ["."]:
            k = self._abs(r)
            if k is None or (k not in self.tree and not self._is_dir(k)):
                out.append(f"find: '{r}': No such file or directory\n")
                continue
            pre = k + "/" if k else ""
            items = {k} | {p for p in self.tree if p.startswith(pre)} | {d for d in self.dirs if d.startswith(pre)}
            items |= {"/".join(p.split("/")[:n]) for p in list(items) for n in range(1, p.count("/") + 1)
                      if "/".join(p.split("/")[:n]).startswith(pre)}
            for it in sorted(items):
                rel = it[len(pre):] if it != k else ""
                d = 0 if not rel else rel.count("/") + 1
                if depth is not None and d > depth:
                    continue
                is_dir = it not in self.tree
                if typ == "f" and is_dir or typ == "d" and not is_dir:
                    continue
                base = (it.split("/")[-1] if it else ".")
                if names and not all(fnmatch.fnmatchcase(base.lower() if ci else base, n.lower() if ci else n)
                                     for n, ci in names):
                    continue
                shown = r.rstrip("/") + ("/" + rel if rel else "") if r not in (".", "./") else ("./" + rel if rel else ".")
                if exe:
                    o, _ = self._dispatch(exe[0], [x.replace("{}", shown) for x in exe[1:]], None)
                    out.append(o)
                else:
                    out.append(shown + "\n")
        return "".join(out), 0

    KNOWN = ("ls cat head tail grep wc pwd cd echo printf sed cp mv rm touch mkdir find which type command stat file od "
             "xxd date env printenv whoami id readlink realpath basename dirname true false chmod sort uniq tr cut awk "
             "bash sh export alias uname md5sum hexdump").split()
    BUILTINS = {"cd", "echo", "pwd", "printf", "type", "command", "true", "false"}

    def _c_which(self, args, stdin):
        out, code = [], 0
        for a in (x for x in args if not x.startswith("-")):
            if a in self.KNOWN and a not in ("cd", "type", "command"):
                out.append(f"/usr/bin/{a}\n")
            else:
                code = 1
        return "".join(out), code

    def _c_type(self, args, stdin):
        out, code = [], 0
        for a in (x for x in args if not x.startswith("-")):
            if a in self.BUILTINS:
                out.append(f"{a} is a shell builtin\n")
            elif a in self.KNOWN:
                out.append(f"{a} is /usr/bin/{a}\n")
            else:
                out.append(f"bash: type: {a}: not found\n")
                code = 1
        return "".join(out), code

    def _c_command(self, args, stdin):
        if args[:1] == ["-v"]:
            return self._c_which(args[1:], stdin)
        return self._dispatch(args[0], args[1:], stdin) if args else ("", 0)

    def _c_stat(self, args, stdin):
        out, code = [], 0
        for a in (x for x in args if not x.startswith("-")):
            k = self._abs(a)
            if k is None or (k not in self.tree and not self._is_dir(k)):
                out.append(f"stat: cannot statx '{a}': No such file or directory\n")
                code = 1
                continue
            size = len(self.tree.get(k, "").encode()) if k in self.tree else 4096
            kind = "regular file" if k in self.tree else "directory"
            out.append(f"  File: {a}\n  Size: {size}\tBlocks: 8\tIO Block: 4096   {kind}\n"
                       f"Access: (0644/-rw-r--r--)  Uid: ( 1000/   agent)   Gid: ( 1000/   agent)\n")
        return "".join(out), code

    def _c_file(self, args, stdin):
        out = []
        for a in (x for x in args if not x.startswith("-")):
            k = self._abs(a)
            if k is not None and k in self.tree:
                t = self.tree[k]
                kind = "empty" if not t else ("Bourne-Again shell script, ASCII text executable" if t.startswith("#!")
                                              else "CSV text" if k.endswith(".csv") else "ASCII text")
            elif self._is_dir(k):
                kind = "directory"
            else:
                kind = "cannot open (No such file or directory)"
            out.append(f"{a}: {kind}\n")
        return "".join(out), 0

    def _c_od(self, args, stdin):
        _, _, rest = self._flags(args)
        text, err = self._text_of(rest, stdin, "od")
        if err:
            return err, 1
        b = text.encode()
        rows = []
        for i in range(0, len(b), 16):
            chunk = b[i:i + 16]
            rows.append(f"{i:07o} " + " ".join(f"{chr(c) if 32 <= c < 127 else {10: chr(92) + 'n', 9: chr(92) + 't'}.get(c, '???'):>3}"
                                                for c in chunk) + "\n")
        return "".join(rows) + f"{len(b):07o}\n", 0

    def _c_xxd(self, args, stdin):
        _, _, rest = self._flags(args)
        text, err = self._text_of(rest, stdin, "xxd")
        if err:
            return err, 1
        b = text.encode()
        return "".join(f"{i:08x}: " + " ".join(b[j:j + 2].hex() for j in range(i, min(i + 16, len(b)), 2)) + "  "
                       + "".join(chr(c) if 32 <= c < 127 else "." for c in b[i:i + 16]) + "\n"
                       for i in range(0, len(b), 16)), 0

    def _c_date(self, args, stdin):
        return DATE + "\n", 0

    def _c_env(self, args, stdin):
        return "".join(f"{k}={v}\n" for k, v in ENV.items()), 0

    def _c_printenv(self, args, stdin):
        if args:
            return "".join(ENV[a] + "\n" for a in args if a in ENV), 0 if all(a in ENV for a in args) else 1
        return self._c_env(args, stdin)

    def _c_whoami(self, args, stdin):
        return "agent\n", 0

    def _c_id(self, args, stdin):
        return "uid=1000(agent) gid=1000(agent) groups=1000(agent)\n", 0

    def _c_readlink(self, args, stdin):
        rest = [a for a in args if not a.startswith("-")]
        k = self._abs(rest[0]) if rest else ""
        return (APP + ("/" + k if k else "") + "\n", 0) if k is not None else ("", 1)

    _c_realpath = _c_readlink

    def _c_basename(self, args, stdin):
        return (args[0].rstrip("/").split("/")[-1] + "\n", 0) if args else ("", 1)

    def _c_dirname(self, args, stdin):
        return (("/".join(args[0].rstrip("/").split("/")[:-1]) or ".") + "\n", 0) if args else ("", 1)

    def _c_true(self, args, stdin):
        return "", 0

    def _c_false(self, args, stdin):
        return "", 1

    def _c_chmod(self, args, stdin):
        return "", 0

    def _c_export(self, args, stdin):
        return "", 0

    def _c_alias(self, args, stdin):
        return "", 0

    def _c_uname(self, args, stdin):
        return ("Linux sandbox 6.8.0 #1 SMP x86_64 GNU/Linux\n" if "-a" in args else "Linux\n"), 0

    def _c_md5sum(self, args, stdin):
        import hashlib

        out, code = [], 0
        for a in (x for x in args if not x.startswith("-")) or [None]:
            text = stdin or "" if a is None else self._read(a)[0]
            if text is None:
                out.append(f"md5sum: {a}: No such file or directory\n")
                code = 1
                continue
            out.append(f"{hashlib.md5(text.encode()).hexdigest()}  {a or '-'}\n")  # noqa: S324  (a checksum, as md5sum)
        return "".join(out), code

    def _c_hexdump(self, args, stdin):
        return self._c_xxd([a for a in args if a != "-C"], stdin)

    def _c_sort(self, args, stdin):
        fl, val, rest = self._flags(args, ("-k", "-t"))
        text, err = self._text_of(rest, stdin, "sort")
        if err:
            return err, 2
        lines = text.splitlines()
        col = int(re.match(r"\d+", val["-k"]).group(0)) - 1 if "-k" in val and re.match(r"\d+", val["-k"]) else None
        sep = val.get("-t")

        def key(ln):
            f = (ln.split(sep) if sep else ln.split())
            v = f[col] if col is not None and col < len(f) else ln
            if "n" in fl:
                m = re.match(r"\s*-?\d+(\.\d+)?", v)
                return float(m.group(0)) if m else 0.0
            return v
        lines.sort(key=key, reverse="r" in fl)
        if "u" in fl:
            lines = list(dict.fromkeys(lines))
        return "".join(ln + "\n" for ln in lines), 0

    def _c_uniq(self, args, stdin):
        fl, _, rest = self._flags(args)
        text, err = self._text_of(rest, stdin, "uniq")
        if err:
            return err, 1
        out, prev, n = [], None, 0
        for ln in text.splitlines() + [None]:
            if ln == prev:
                n += 1
                continue
            if prev is not None:
                out.append((f"{n:>7} " if "c" in fl else "") + prev + "\n")
            prev, n = ln, 1
        return "".join(out), 0

    def _c_tr(self, args, stdin):
        fl, _, rest = self._flags(args)
        text = stdin or ""
        def expand(s):
            s = s.replace("[:upper:]", "A-Z").replace("[:lower:]", "a-z").replace("\\n", "\n")
            return re.sub(r"(.)-(.)", lambda m: "".join(chr(c) for c in range(ord(m.group(1)), ord(m.group(2)) + 1)), s)
        if "d" in fl and rest:
            return "".join(c for c in text if c not in expand(rest[0])), 0
        if len(rest) >= 2:
            a, b = expand(rest[0]), expand(rest[1])
            return text.translate({ord(x): (b[i] if i < len(b) else b[-1]) for i, x in enumerate(a)}), 0
        return text, 0

    def _c_cut(self, args, stdin):
        fl, val, rest = self._flags(args, ("-d", "-f", "-c"))
        text, err = self._text_of(rest, stdin, "cut")
        if err:
            return err, 1
        def pick(spec, n):
            out = set()
            for part in spec.split(","):
                a, _, b = part.partition("-")
                lo, hi = int(a or 1), int(b) if b else (int(a) if not _ else n)
                out |= set(range(lo, hi + 1))
            return sorted(out)
        out = []
        for ln in text.splitlines():
            if "-c" in val:
                out.append("".join(ln[i - 1] for i in pick(val["-c"], len(ln)) if i <= len(ln)))
            else:
                d = val.get("-d", "\t")
                fs = ln.split(d)
                out.append(d.join(fs[i - 1] for i in pick(val.get("-f", "1"), len(fs)) if i <= len(fs)))
        return "".join(x + "\n" for x in out), 0

    # ---- awk, the subset the pilot used -----------------------------------------------------------------------------
    def _c_awk(self, args, stdin):
        fs, rest, it = None, [], iter(args)
        for a in it:
            if a == "-F":
                fs = next(it, " ")
            elif a.startswith("-F"):
                fs = a[2:]
            else:
                rest.append(a)
        if not rest:
            return "awk: no program\n", 2
        prog, files = rest[0], rest[1:]
        text, err = self._text_of(files[:1], stdin, "awk")
        if err:
            return err, 2
        try:
            return _awk(prog, text, fs), 0
        except Exception as e:  # noqa: BLE001  (an awk the subset cannot read answers as awk would, with an error)
            return f"awk: syntax error ({type(e).__name__})\n", 2


def _awk(prog: str, text: str, fs: str | None) -> str:
    """BEGIN/END blocks, `cond { stmts }` rules, `+=`, `=`, `print` and `printf`, `$N`, `NR`, `NF`."""
    blocks = re.findall(r"(BEGIN|END|[^{}]*?)\s*\{([^{}]*)\}", prog)
    if not blocks:
        blocks = [("", "print $0")]
    env: dict = {}
    out: list[str] = []

    def val(expr: str, fields: list[str], nr: int) -> object:
        e = re.sub(r"\$(\d+|NF)", lambda m: f"_f({m.group(1) if m.group(1) != 'NF' else len(fields) - 1})", expr)
        e = re.sub(r"\bNR\b", str(nr), e)
        e = re.sub(r"\bNF\b", str(len(fields) - 1), e)
        e = e.replace("&&", " and ").replace("||", " or ").replace("!", " not ").replace(" not =", "!=")

        def _f(i):
            s = fields[i] if i < len(fields) else ""
            try:
                return float(s) if re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", s) else s
            except ValueError:
                return s
        names = {k: v for k, v in env.items()}
        for w in set(re.findall(r"\b[A-Za-z_]\w*\b", e)) - {"_f", "and", "or", "not"} - set(names):
            names[w] = 0
        return eval(e, {"__builtins__": {}}, names | {"_f": _f})  # noqa: S307  (a restricted awk expression)

    def fmt(v):
        return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)

    def run(stmts: str, fields: list[str], nr: int) -> None:
        for s in [x.strip() for x in re.split(r";|\n", stmts) if x.strip()]:
            m = re.fullmatch(r"(\w+)\s*(\+|-)?=\s*(.+)", s)
            if s.startswith("printf"):
                parts = [p.strip() for p in s[6:].strip().strip("()").split(",")]
                f, args = parts[0].strip('"').replace("\\n", "\n"), [val(p, fields, nr) for p in parts[1:]]
                out.append(re.sub(r"%[-\d.]*[sdf]", lambda mm: (f"{{:{mm.group(0)[1:-1]}{mm.group(0)[-1]}}}"
                                                                .format(args.pop(0)) if args else ""), f))
            elif s.startswith("print"):
                body = s[5:].strip()
                vals = [fmt(val(p, fields, nr)) for p in body.split(",")] if body else [fields[0]]
                out.append(" ".join(vals) + "\n")
            elif m:
                v = val(m.group(3), fields, nr)
                cur = env.get(m.group(1), 0)
                env[m.group(1)] = v if not m.group(2) else (cur + v if m.group(2) == "+" else cur - v)

    for cond, stmts in blocks:
        if cond.strip() == "BEGIN":
            run(stmts, [""], 0)
    for nr, ln in enumerate(text.splitlines(), 1):
        fields = [ln] + (ln.split(fs) if fs else ln.split())
        for cond, stmts in blocks:
            c = cond.strip()
            if c in ("BEGIN", "END"):
                continue
            if not c or val(c, fields, nr):
                run(stmts, fields, nr)
    for cond, stmts in blocks:
        if cond.strip() == "END":
            run(stmts, [""], 0)
    return "".join(out)
