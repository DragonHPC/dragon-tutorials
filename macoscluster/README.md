# Building a Two-Node Mac OS HPC Cluster with Dragon

As of the mid-October 2026 release of 0.15.0, Dragon supports multi-node Mac OS
— which means Mac Studios, Macbook Pros, Mac Minis, and even Macbook Airs can be
configured to be a genuine cluster.

This post walks through everything needed to get there: shared storage, keeping
the nodes alive, the network. Every command here was run on a working cluster.

## The hardware

In this post we are configuring a two node cluster consisting of:

| role | machine | cores | memory | network |
|---|---|---|---|---|
| head + compute | Mac Studio (M3 Ultra) | 28 logical / 20 performance | 96 GB | built-in 2.5GbE |
| compute | MacBook Pro 16" (M3 Max) | 16 logical / 12 performance | 48 GB | 2.5GbE via Thunderbolt adapter |

Both running Mac OS 27.0, Python 3.12.10, `dragonhpc` 0.15.0.

144 GB of unified memory across the two nodes, and the nodes need not be
identically specified — Dragon is happy with a heterogeneous cluster, though you
will want to size the work you place on each node accordingly.

Check what you have with:

```bash
sysctl -n hw.memsize hw.logicalcpu
```

This will report the total number of bytes of memory and the number of CPU cores
for your hardware. Nothing in this post requires specific to models. Any two or
more Macs with Thunderbolt ports will do.


## Step 1: Matching user accounts on every node

Do this before anything else. Passwordless SSH, the shared filesystem and Dragon
itself all assume the same account exists on every node. It's best to set this
up correctly right away.

Create an account with the **same username on every node**. That part is
essential: Dragon connects to backend nodes over SSH without specifying a user,
so `ssh node2` must land on the right account, and home-directory paths need to
look the same everywhere.

### Matching UIDs: not required, but convenient

Behind each username is a numeric UID, and NFS tracks ownership by **number**,
not by name. These do *not* have to match — the cluster used for this post runs
happily with mismatched numbers:

```bash
$ ssh my-studio.local id
uid=501(username) gid=20(staff)
$ ssh my-max.local id
uid=502(username) gid=20(staff)
```

Mac OS assigns the first account on a machine UID 501, the second 502, and so
on, so a mismatch is easy to end up with through nothing but the order you set
the machines up in.

What matching UIDs buy you is simplicity. When the numbers agree, a file created
on any node is owned by *you* on every node, and default permissions just work.
When they don't, every node still sees the file but attributes it to a different
owner, so the shared filesystem has to be permissive enough for a
differently-numbered user to read and write it. That's entirely workable — it
just means permissions become something you have to think about rather than
something that takes care of itself. See
[If the UIDs can't match](#if-the-uids-cant-match) in the next step.

Check what you have:

```bash
for host in my-studio.local my-max.local; do
    printf '%-22s ' "$host"; ssh "$host" id
done
```

If they already agree, there's nothing to do. If they differ and you'd rather
align them, the next section shows how — but skipping it is a perfectly
reasonable choice.

### Making the UIDs match (optional)

If you want them aligned, change one. Pick the node with the fewest files to
rewrite — usually a compute node rather than the file server.

The GUI route: **System Settings → Users & Groups**, right-click the user →
**Advanced Options** → edit **User ID**. You must be logged in as a *different*
admin account; Mac OS will not let you renumber the account you are using.

From the command line, as another admin, with the target user logged out:

```bash
# Inspect the current value
dscl . -read /Users/username UniqueID

# Change it
sudo dscl . -change /Users/username UniqueID 502 501

# Rewrite ownership of everything the old UID owned
sudo find /Users/username -user 502 -exec chown 501 {} \;
sudo find /private/var -user 502 -exec chown 501 {} \; 2>/dev/null
```

Reboot, then confirm with `id -u`. Do not skip the `chown` step — renumbering
the account without it leaves the user unable to read their own home directory.

Match the **group** too if it differs. On modern Mac OS everyone lands in
`staff` (GID 20), so this is usually already consistent.

### Many nodes: use a directory service

Renumbering by hand is fine for two or three machines and miserable beyond that.
For a larger cluster, centralize the accounts so every node gets its identity
from one source.

Mac OS clients can bind to a directory server with:

```bash
# LDAP
sudo dsconfigldap -a ldap.example.com

# Active Directory
sudo dsconfigad -a MACHINE -domain example.com -u adminuser
```

Verify the node is using it:

```bash
dscl localhost -list /LDAPv3/ldap.example.com/Users
```

Apple's own Open Directory server was part of Mac OS Server, which is
discontinued — so in practice you'd point Mac OS clients at an existing LDAP or
Active Directory service, or run OpenLDAP on a Linux box. There is no
Apple-supported way to run the server side on current Mac OS.

For a small lab, leaving the UIDs as they fall and setting permissions
explicitly is perfectly workable — a directory service is worth the trouble when
you are managing accounts across many machines, not merely to align two numbers.

## Step 2: The network

Put every node on the same switch and the same subnet. The goal is **zero hops**
between nodes — every router in the path adds latency you'll pay for on every
message.

A 2.5GbE switch is a good value point for Macs: recent models have 2.5GbE built
in or available via a Thunderbolt adapter, and 2.5GbE switches are cheap.

Check for extra hops:

```bash
traceroute -n 192.168.4.173
```

You want exactly one line — the destination itself. If you see a gateway in
between, your nodes are on different subnets and traffic is being routed rather
than switched.

Confirm the negotiated link speed:

```bash
ifconfig en0 | grep media
# media: autoselect (2500Base-T <full-duplex,flow-control>)
```

If it differs from what you think you have, you have a cable or switch port
limiting you.

### Pin the addresses

DHCP is fine, but a node whose address changes after a reboot breaks your
hostfile. Rather than configuring static IPs on each machine, reserve them in
your router's DHCP server by MAC address — the nodes keep using DHCP, but always
get the same address.

Get each MAC with:

```bash
ifconfig en0 | awk '/ether/{print $2}'
```

You can go to your router software and typically configure your router to
reserve static IPs for the nodes that match your MAC addresses.

## Step 3: A shared filesystem

Every node must see the same files at the same path. Dragon ships your program
to the backend nodes by *path*, not by copying it, so if
`/Volumes/Shared/app.py`, for example, doesn't resolve identically everywhere,
your run dies immediately.

Mac OS supports both NFS and SMB. NFS is the better choice here: it stays up, it
remounts cleanly after a reboot, and it doesn't prompt for credentials.

> **DISCLAIMER - Keep this behind your firewall.** The setup below trades access
> control for convenience: NFS exports authenticated only by source IP, no
> credentials on mount, and — if your UIDs differ — world-readable and
> world-writable files. That is reasonable on a private, trusted network. It is
> not safe anywhere else.
> 
> Put the file server on a machine that has **no route in from the internet**.
> Do not port-forward NFS (TCP/UDP 111 and 2049) through your router, do not run
> it on a cloud instance with a public address, and do not export to a network
> you share with guests or untrusted devices. Anyone who can reach port 2049 can
> mount the export as described here, and anyone who can mount it can read and
> write everything in it — including the code Dragon is about to execute on
> every node.
> 
> If you need a cluster spanning untrusted networks, this is the wrong design;
> put the nodes on a VPN or private subnet first, then follow steps to create a
> secure shared file system. Those directions are not provided here.
> 
> The directions below assume you don't have matching UIDs for your accounts. If
> you do, the permissions can be more restrictive when exporting the file
> system. Use an AI model to help you in that case to get the right permissions.

### On the head node (the file server)

Create the directory and export it. On Mac OS the export file is `/etc/exports`:

```bash
mkdir -p ~/Shared
sudo tee /etc/exports <<'EOF'
/Users/username/Shared -alldirs -maproot=nobody -network 192.168.4.0 -mask 255.255.255.0
EOF
```

Breaking that down:

- The network address should be your network. The 0 means everyone on the
  192.168.4 subnet will have access.
- `alldirs` allow mounting any subdirectory
- `maproot=nobody` means a remote `root` is squashed to `nobody`; standard
  safety
- `network` and `mask` restrict exports to your LAN subnet only

That `network` restriction means only others on your network can reach the
machine, so keep it as narrow as your addressing allows. Exporting without it,
or with a mask wide enough to cover a guest network, publishes the share to
every host that can route to the server.

Then start the NFS server and tell it to come back after a reboot:

```bash
sudo nfsd enable
sudo nfsd start
sudo nfsd checkexports   # syntax-checks /etc/exports; silence means success
```

Verify the export is visible:

```bash
showmount -e localhost
```

### On each compute node

```bash
sudo mkdir -p /Volumes/Shared
sudo mount -t nfs -o nodev,nosuid 192.168.4.51:/Users/username/Shared /Volumes/Shared
```

Confirm it took:

```bash
mount | grep Shared
# 192.168.4.51:/Users/username/Shared on /Volumes/Shared (nfs, nodev, nosuid, mounted by username)
```

To make it survive a reboot, add an `autofs` entry rather than relying on
Finder's "Connect to Server", which only reconnects after someone logs in:

```bash
echo '/Volumes/Shared -fstype=nfs,nodev,nosuid nfs://192.168.4.51/Users/username/Shared' | sudo tee /etc/auto_nfs
echo '/-  auto_nfs' | sudo tee -a /etc/auto_master
sudo automount -vc
```

### If the UIDs can't match

Sometimes you can't renumber — a shared machine, a managed laptop, an account
you don't control. In that case the shared filesystem must be permissive enough
that a differently-numbered user can still work:

```bash
# Source files must be world-readable
chmod 644 /Volumes/Shared/*.py

# Output directories must be world-writable, because workers on the
# other node write there under a different mapped UID
chmod 777 /Volumes/Shared/results
```

`777` is doing exactly what it looks like: granting every user on every machine
that can mount the share full control of that directory. That is acceptable on
an isolated private network and nowhere else — which is the other reason to keep
the file server firewalled off from the outside world.

A directory created with the default `755` works perfectly for workers on the
node that made it and fails for workers anywhere else — so the problem only
surfaces when Dragon happens to place a rank on a separate node.

```
Python: can't open file '/Volumes/Shared/halo3.py': [Errno 13] Permission denied
```

Watch out for `cp` preserving a restrictive mode. A file copied in as `600` is
invisible to every other node, and the error points at Python rather than at the
copy that caused it. After adding source to the share:

```bash
chmod 644 /Volumes/Shared/*.py
xattr -c /Volumes/Shared/*.py   # strip quarantine attributes too
```

None of this is a workaround for a broken setup — it is simply what you trade
for not aligning the UIDs. Matching UIDs lets default permissions take care of
themselves; mismatched UIDs mean you set the modes deliberately. Both run just
fine.


## Step 4: Stop the nodes going to sleep

You can disable sleep on every node with these commands. The commands will help
you keep tcp connections open which would be needed for long running Dragon
programs and to keep ssh sessions connected if you are opening an ssh connection
to a node.

```bash
sudo pmset -a sleep 0          # never system-sleep
sudo pmset -a disksleep 0      # never spin down storage
sudo pmset -a powernap 0       # no background wake/sleep cycling
sudo pmset -a womp 1           # wake on network access
sudo pmset -a tcpkeepalive 1   # keep TCP alive through display sleep
```

Letting the *display* sleep is fine and saves power:

```bash
sudo pmset -a displaysleep 10
```

On Apple silicon laptops, also enable High Power Mode so the machine doesn't
throttle during long runs:

```bash
sudo pmset -a powermode 2
```

Verify:

```bash
pmset -g custom
```

You want `sleep 0`, `disksleep 0`, `powernap 0` under **both** "AC Power" and
"Battery Power". Closing the lid will still suspend a laptop unless it's on
external power with a display attached — so for a cluster node, keep it plugged
in and lid-open, or use `caffeinate` as a belt-and-braces measure:

```bash
caffeinate -dimsu &
```

## Step 5: SSH that doesn't drop

Dragon launches backend services over SSH and keeps those connections open for
the life of the run. A dropped connection kills the job.

First, enable Remote Login on every node (System Settings → General → Sharing →
Remote Login), or from the terminal:

```bash
sudo systemsetup -setremotelogin on
```

### Passwordless login between *every pair* of nodes

Dragon cannot answer a password prompt, so key-based auth has to be in place
before it runs. The part that catches people: it is not enough to set this up
from the head node outwards. Nodes connect to **each other**, so every node must
be able to reach every other node — and itself — without a prompt. On a two-node
cluster that's four directions, not one.

The simplest arrangement for a small cluster is a single key pair shared by all
nodes, with each node trusting that key. Generate it once on the head node:

```bash
ssh-keygen -t ed25519            # if you don't already have a key
```

Authorize it on the head node itself, then copy both the key pair and the
authorization to every other node:

```bash
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys

for host in my-max.local; do          # add each additional node here
    ssh-copy-id "$host"               # trust the key there
    scp ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub "$host":~/.ssh/
done
```

Copying the private key to each node is what lets *them* initiate connections
too. If you would rather not have the same private key on every machine,
generate a separate pair per node and append each public key to every node's
`authorized_keys` — more keys to manage, same result.

Now verify **all** directions, including each node to itself:

```bash
for src in my-studio.local my-max.local; do
    for dst in my-studio.local my-max.local; do
        printf '%-18s -> %-18s ' "$src" "$dst"
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$src" \
            "ssh -o BatchMode=yes -o ConnectTimeout=5 $dst true" \
            && echo OK || echo FAIL
    done
done
```

`BatchMode=yes` disables every prompt, so this reports `FAIL` under exactly the
conditions Dragon would hang on. Every cell must read `OK` before you continue.

Node-to-itself matters too: Dragon may launch services on the head node through
the same SSH path it uses for the others, and a missing self-entry fails in the
same silent way.

### Accept every host fingerprint before you run Dragon

On a first connection SSH prompts:

```
The authenticity of host 'my-max.local (192.168.4.173)' can't be established.
ED25519 key fingerprint is SHA256:...
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Dragon launches its backends non-interactively. There is nobody to type `yes`,
so the connection simply never completes and the run hangs at startup with no
useful diagnostic.

SSH keys host entries **per name**, not per machine. `my-max.local`, `my-max`
and `192.168.4.173` are three separate entries even though they're one computer.
And `known_hosts` is per *client*, so accepting a fingerprint on one node does
nothing for any other node.

That means the same all-pairs rule as the keys: **every node needs a saved
fingerprint for every node, itself included, under every name you might use.**
Do not try to collect them by hand.

`ssh-keyscan` fetches and saves host keys without any interactive prompt, so the
whole matrix can be seeded in one pass. Run this **on every node**:

```bash
NAMES="my-studio.local my-max.local 192.168.4.51 192.168.4.173"
ssh-keyscan -H $NAMES >> ~/.ssh/known_hosts
sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts   # drop duplicates from re-runs
```

Or drive it from one node, so you only type it once:

```bash
NAMES="my-studio.local my-max.local 192.168.4.51 192.168.4.173"
for node in my-studio.local my-max.local; do
    ssh "$node" "ssh-keyscan -H $NAMES >> ~/.ssh/known_hosts; \
                 sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts"
done
```

If you prefer to connect interactively and answer `yes` each time, the
equivalent is a nested loop — every source node to every destination name:

```bash
for src in my-studio.local my-max.local; do
    for dst in $NAMES; do
        ssh "$src" "ssh -o ConnectTimeout=5 $dst true"
    done
done
```

Then confirm nothing is left to prompt, in both directions for every pair:

```bash
for src in my-studio.local my-max.local; do
    for dst in my-studio.local my-max.local; do
        printf '%-18s -> %-18s ' "$src" "$dst"
        ssh -o BatchMode=yes "$src" \
            "ssh -o BatchMode=yes -o ConnectTimeout=5 $dst true" \
            && echo OK || echo FAIL
    done
done
```

`BatchMode=yes` disables all prompting, so any missing fingerprint or key shows
up as `FAIL` rather than as a hung Dragon run. Every line must read `OK`.

You may need to repeat this later if you set up the Thunderbolt bridge. Tßhe
Thunderbolt bridge in Step 7 introduces a second set of names and addresses, and
those are separate `known_hosts` entries again.

Then stop idle connections being reaped. On the **client** side, in
`~/.ssh/config`:

```
Host my-*
    ServerAliveInterval 30
    ServerAliveCountMax 120
    TCPKeepAlive yes
```

And on the **server** side, in `/etc/ssh/sshd_config`:

```
ClientAliveInterval 30
ClientAliveCountMax 120
TCPKeepAlive yes
```

```bash
sudo launchctl kickstart -k system/com.openssh.sshd
```

That combination tolerates an hour of silence before giving up, which is enough
for a long simulation that produces no output while it runs.

## Step 6: Grant Python network and disk permission

Mac OS sandboxes applications far more aggressively than Linux, and the Python
interpreter is just another application as far as the OS is concerned. Two
permissions matter, and neither is granted by default.

### Local Network

Since Mac OS Sequoia, an application must be explicitly authorized before it can
talk to other machines on the local network. Without it, Python's sockets fail
or time out when connecting to another node — while `ping` from the terminal
works perfectly, which makes it look like an application bug rather than a
permissions problem.

Go to **System Settings → Privacy & Security → Local Network** and enable the
entry for Python. You may also need to enable Terminal (or iTerm, or whichever
terminal you launch Dragon from), since a process can inherit its parent's
sandbox decision.

The trap: the entry only appears in that list **after** the application has
tried to use the local network at least once. If you don't see Python there, run
something that touches the network first:

```bash
python3 -c "import socket; socket.create_connection(('192.168.4.173', 22), timeout=2)"
```

Then look again.

If you installed Python from python.org, the entry is listed as **Python**; with
Homebrew it may appear as **python3**. Either way, check which interpreter you
are actually running, because granting permission to the wrong one is a
frustrating way to lose an hour:

```bash
source /Volumes/Shared/env3.12/bin/activate
python3 -c "import sys; print(sys.executable); print(sys.base_prefix)"
```

A virtual environment's `python3` is a symlink to the base interpreter, and it's
the **base** interpreter that Mac OS tracks.

### Full Disk Access

Dragon reads and writes shared storage, temporary files and logs. If your shared
mount lives somewhere Mac OS considers protected — anything under `Desktop`,
`Documents`, `Downloads`, or an external/network volume — add Python under
**System Settings → Privacy & Security → Full Disk Access** too.

The symptom is a `PermissionError` on a path that `ls` from your shell can read
perfectly well, because your shell has permission and the interpreter doesn't.

Do this on **every node**. Settings are per-machine, and a cluster where one
node lacks Local Network permission fails in a way that looks exactly like a
network misconfiguration.

## Optional Step 7: Thunderbolt bridge — worth the cable

Two Macs connected by a single Thunderbolt cable form an IP network with far
more bandwidth than a 2.5GbE switch and modestly lower latency. Whether that
translates into a faster application depends entirely on what your workload is
bound by — the numbers below are specific about which.

Connect a **Thunderbolt 3/4 or USB4 cable** directly between the two machines.
This must be a real data cable — a USB-C charging cable looks identical and will
not work.

Mac OS creates the bridge interface automatically. Give each end an address on a
subnet of its own, separate from your LAN:

```bash
# head node
sudo networksetup -setmanual "Thunderbolt Bridge" 10.10.10.1 255.255.255.0 ""
# compute node
sudo networksetup -setmanual "Thunderbolt Bridge" 10.10.10.2 255.255.255.0 ""
```

Note the **four** arguments. The `setmanual` option requires a router argument
even when there isn't one; pass an empty string. Omitting it fails with a bare
usage message and exit code 1.

Enable jumbo frames on both ends:

```bash
sudo networksetup -setMTU bridge0 9000
```

Verify the link is actually up — and be careful here, because the flags lie:

```bash
ifconfig bridge0 | grep -E 'status:|inet '
# inet 10.10.10.1 netmask 0xffffff00 broadcast 10.10.10.255
# status: active
```

`bridge0` always exists on any Mac with Thunderbolt ports, and its flags always
read `UP,RUNNING` even with nothing plugged in. **`status:` is the only field
that reflects a real cable.** Check `system_profiler SPThunderboltDataType` to
confirm the peer is the other Mac and not a dock.

Add the bridge addresses to `/etc/hosts` on every node:

```
10.10.10.1  studio-tb
10.10.10.2  max-tb
```

### Seed SSH for the new names

These names and addresses are new to SSH, and `known_hosts` entries are per
name. Dragon will connect to the nodes as `studio-tb` and `max-tb`, so repeat
the all-pairs seeding from Step 4 for the bridge names before running anything:

```bash
TB_NAMES="studio-tb max-tb 10.10.10.1 10.10.10.2"
for node in my-studio.local my-max.local; do
    ssh "$node" "ssh-keyscan -H $TB_NAMES >> ~/.ssh/known_hosts; \
                 sort -u ~/.ssh/known_hosts -o ~/.ssh/known_hosts"
done
```

Then verify every direction over the bridge names:

```bash
for src in studio-tb max-tb; do
    for dst in studio-tb max-tb; do
        printf '%-10s -> %-10s ' "$src" "$dst"
        ssh -o BatchMode=yes "$src" \
            "ssh -o BatchMode=yes -o ConnectTimeout=5 $dst true" \
            && echo OK || echo FAIL
    done
done
```

Skipping this could produce a Dragon run that hangs at startup with no error,
having worked perfectly over the LAN earlier.

### What it buys you

Measured on the cluster above, single TCP stream:

| path | bandwidth | RTT |
|---|---|---|
| 2.5GbE switch | 2.30 Gbit/s | 0.508 ms |
| Thunderbolt bridge | 37.47 Gbit/s | 0.389 ms |

What this means:

- **Bandwidth: 16x.** 37.47 Gbit/s is 94% of the Thunderbolt 4 line rate, so Mac
  OS Thunderbolt-IP is not leaving much on the table. The switch result, 2.30
  Gbit/s, is likewise 92% of 2.5GbE — nothing wrong with it, there is just far
  less headroom to begin with.
- **Latency: 1.3x.** 0.508 ms to 0.389 ms. A real improvement, but nothing like
  the bandwidth figure.

So "16x faster network" would be a fair description of the *link* and a
misleading description of your *job*. How much of that 16x reaches your
application depends on what it spends its time doing.

#### One example: a halo-exchange solver

To give the numbers something concrete to attach to, here is one workload we
happened to be running — a lattice-Boltzmann fluid solver, 16 ranks across both
nodes, exchanging boundary data every timestep. Nothing about this post requires
you to care about CFD; it is simply the application that was to hand, and it
sits at one extreme of the spectrum.

Switching it from the switch to the Thunderbolt bridge took wall time from 22.9
s to 20.3 s — about **11% faster**, with the exchange portion dropping 23%.

Why so far short of 16x? This solver sends 8 KB per rank per timestep. At that
size you are paying per-message latency, not moving bulk data, and latency is
the axis Thunderbolt barely improves. It is close to a worst case for this
upgrade but still an improvement over 2.5Gb/s Ethernet.

#### Where your workload might land

| if your application mostly... | expect |
|---|---|
| exchanges small messages frequently (tightly-coupled solvers, fine-grained collectives) | modest gains — latency-bound, like the example above |
| moves large blocks (array transfers, checkpointing, shuffling data between stages) | much closer to the 16x |
| barely communicates (ensembles, parameter sweeps, embarrassingly parallel work) | little or nothing — the network was never the limit |

Measure before assuming. Dragon ships `p2p_lat.py`, which reports latency across
a range of message sizes and will tell you quickly which regime you are in.

Even at the unflattering end, 11% for the price of one cable is worth having —
just don't size your expectations from the bandwidth headline.

### Don't bother with a second cable

We tried it. A second Thunderbolt cable between the same two machines gained
nothing measurable (20.4 s vs 20.7 s, inside run-to-run noise) with the halo
application. Dragon will make use of multiple network interfaces if they exist.
But the connections are assumed to be many to many and a second bridge cable is
not many to many with other network interfaces.

## Step 8: Install Dragon

After making sure every node has Python installed and permissions have been
granted for network and disk acess, create one virtual environment **on the
shared filesystem**, so every node runs the same interpreter and the same
packages from the same path:

```bash
cd /Volumes/Shared
python3 -m venv env3.12
source env3.12/bin/activate
pip install --upgrade pip
pip install dragonhpc
```

Verify on every node:

```bash
python3 -V          # Python 3.12.10
source env3.12/bin/activate
pip show dragonhpc  # Version: 0.15.0
```

Create a hostfile listing the nodes by their network names. For **Thunderbolt**
use names like these:

```bash
printf 'studio-tb\nmax-tb\n' > /Volumes/Shared/hostfile
```

Create a small test program, used by both of the next two steps. Save it as
`/Volumes/Shared/hello.py`:

```python
import dragon.native as dn
from socket import gethostname


def worker():
    print(f"Hello from {gethostname()}", flush=True)


def main():
    procs = [dn.Process(target=worker, args=()) for _ in range(6)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
```

`dragon.native.Process` is Dragon's own process type, which the runtime is free
to place on any node in the allocation. By default Dragon will use round-robin
placement across the nodes of the cluster so printing the hostname tells you
whether Dragon is configured correctly to use the cluster.

If your UIDs differ across nodes, remember to make it readable everywhere:

```bash
chmod 644 /Volumes/Shared/hello.py
```

## Step 9: Tell Dragon which network to use

Dragon discovers every interface on each node and picks the first one that
matches a regular expression. The default pattern matches essentially every
interface a Mac has. Mac OS has interfaces starting with `bridge` and `en`.

```python
DEFAULT_TRANSPORT_NETIF = r"^(hsn|ipogif|ib|eth|bridge|en)\w+$"
```

Interface enumeration order differs between machines, so each node can choose a
*different* network — and Dragon pairs them positionally. On our cluster the
default picked a link-local `169.254.x` address on an **unplugged** adapter for
one node and the Thunderbolt address for the other, then tried to pair them.

The result was intermittent startup hangs. The fix is to tell Dragon that you
prefer one network interface over another.

```bash
dragon --hostfile hostfile --network-prefix '^bridge' myapp.py
```

`'^bridge'` matches `bridge0` and any later `bridgeN`. The regex is applied with
`re.match`, which anchors at the start with the use of `^`. If you are using the
regular network and not a bridge connection, then specify `^en` as the network
prefix.

Alternatively, you can create a script to run your application:

```bash
#!/bin/bash
NETPREFIX='^bridge'
dragon --hostfile hostfile --network-prefix "$NETPREFIX" myapp.py
```

Getting the hostfile names right is not sufficient — those only control which
host SSH may connect to. Setting the network interface card prefix guarantees
that Dragon will choose the interfaces you want to use.

### Checking which interface each node actually chose

Dragon writes nothing to disk by default, so to see the selection you have to
ask for logging. The relevant message comes from the backend actor on each node,
which writes to the per-actor log file:

```bash
dragon --hostfile hostfile --network-prefix '^bridge' -l DEBUG hello.py
```

`dragon -l DEBUG` is commonly used for debugging any issues while running Dragon
programs.

That run drops an `LA_BE_<hostname>_<timestamp>.log` per node in the working
directory. Each should report exactly one address, on the bridge:

```bash
grep -h 'Found IP addresses' LA_BE_*.log
# ... LA_BE.get_host_info ... Found IP addresses: 10.10.10.1
# ... LA_BE.get_host_info ... Found IP addresses: 10.10.10.2
```

If a node lists several addresses, your prefix is too loose and Dragon is
choosing positionally among them. If two nodes report addresses on *different*
subnets, they may be paired with each other — which is the misconfiguration this
step prevents.

Turn logging back off for production runs. Using the `DEBUG` option produces a
lot of output on the shared filesystem.

## Step 10: Run something

Start with the `hello.py` from Step 8 — a smoke test that proves processes
really are landing on both machines:

```bash
cd /Volumes/Shared
source env3.12/bin/activate
dragon --hostfile hostfile --network-prefix '^bridge' hello.py
```

You should see both hostnames, roughly evenly split:

```
Hello from my-studio.local
Hello from my-studio.local
Hello from my-studio.local
Hello from my-max.local
Hello from my-max.local
Hello from my-max.local
```

If every line names the same host, the run is not actually distributed — check
your hostfile and that the backend came up on the second node.

Then a real benchmark. Dragon ships `gups_ddict.py`, which hammers the
distributed dictionary and is a good end-to-end test of the interconnect. You
will need to copy this program from the
[Dragon repository](https://github.com/DragonHPC/dragon). You can copy the
`gups_ddict.py` program from the `examples/benchmarks` directory.

```bash
dragon --hostfile hostfile --network-prefix '^bridge' gups_ddict.py --cluster --cluster_min_nodes=2
```

For a pure latency measurement you can grab the `p2p_lat.py` program from the
`examples/multiprocessing` directory.

```bash
dragon --hostfile hostfile --network-prefix '^bridge' --nodes 2 \
    p2p_lat.py --iterations 100 --lg_max_message_size 14
```

On our cluster that reports ~250 µs for messages up to 1 KB, rising to ~292 µs
at 8 KB. Note how flat that is: message *size* barely matters, message *count*
dominates.

**Don't trust `ping` for latency.** Mac OS services ICMP on a slower path than
established TCP. We measured 0.8 ms by ping on a link where Dragon's own
`p2p_lat` reported 259 µs. Use the application-level benchmark.

## Troubleshooting

**Cleaning up after a crash.** Dragon allocates POSIX shared memory. A run
killed with `SIGKILL` never unlinks it, and because Mac OS has no `/dev/shm`,
leaked segments can't be enumerated. Use:

```bash
dragon-cleanup --hostfile hostfile
```

Note the `--hostfile` — without it, cleanup only runs on the local node.

**Diagnosing a hang.** Mac OS writes crash reports to
`~/Library/Logs/DiagnosticReports/*.ips`. If a "hang" leaves one behind, it's
actually a crash:

```bash
ls -lt ~/Library/Logs/DiagnosticReports/ | head
```

If you see **`status: inactive` on a bridge with correct flags.** The cable
isn't seated, or it's a charge-only USB-C cable.

## Final Thoughts

A two-node cluster, assembled from machines you already own, running real
distributed workloads over a 40 Gb/s interconnect is pretty cool. For the one
application we benchmarked here, the two nodes together beat the best
single-node configuration, and exceeded the aggregate memory bandwidth ceiling
of either machine alone — which is the point of adding a node in the first
place. Whether your own workload scales that way is an empirical question, but
support for Mac OS clusters is now there to answer it!

The parts that actually mattered, in order:

1. `--network-prefix` — without it, nothing else is reliable
2. SSH fingerprints and Local Network permission — both fail silently, and both
   look like network bugs rather than the permission prompts they are
3. The Thunderbolt bridge — 16x the bandwidth of 2.5GbE but only 1.3x the
   latency, which bought this particular workload 11%
4. NFS permissions — the failure mode is obscure and intermittent
5. Sleep settings — a sleeping node is a failed run

And one that isn't about performance at all: this design assumes a private,
trusted network throughout. The shared filesystem has no real access control, so
keep the file server firewalled off from the internet and treat anything that
can reach it as having full access to your code and data.
