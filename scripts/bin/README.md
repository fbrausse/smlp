# Virtual display, GUI support and data sharing for Docker containers 

## Overview

### Validated Ubuntu environments

- `Ubuntu 24.04` 
- `Windows 11/WSL2` with `Ubuntu 24.04` installed and [wslg](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) enabled

**Data sharing**

All `enter*` commands mentioned below create `$HOME/shared` directory and mount it to Docker container `/shared` directory

**Docker Hub Images Details:**

| Repository | Tag | Image Index Digest |
|--------------------------------------------------|-------|-------------|
| mdmitry1/smlp-test-build-almalinux\_9-python311   | latest | sha256:a06cdf5fe23e |
| mdmitry1/smlp-test-build-opensuse\_15.5-python311 | latest | sha256:42693e6e9eaa |
| mdmitry1/smlp-test-build-ubuntu\_22.04-python311  | latest | sha256:15f41fd25dfa |
| mdmitry1/python311-dev                            | latest | sha256:53f8f6f4455a |

GUI and data sharing is supported for container `mdmitry1/python311-dev:latest` only.
Virtual display is supported for all containers.

## Quick Start

### Run Docker containers using virtual display

Run container:

```bash
enter_container [<image_name>[:<tag>]]
```

Default container is: `mdmitry1/python311-dev:latest`

In the container run commands:

```bash
/app/open_virtual_display
export DISPLAY=:99
```

Check SMLP installation 

```bash
smlp -h
```

### Run mdmitry1/python311-dev container with VNC client

- Recommended VNC servers:
  - Windows: RealVNC®
  - Ubuntu 24.04: remmina

[RealVNC® installation instructions](https://github.com/SMLP-Systems/smlp/blob/master/doc/RealVNC.md)

- Remmina installation command

```bash
sudo apt install remmina
```

Configure `remmina` as shown below:

![Remmina Profile](media/remmina.png)<br><br>


Run container:

```bash
enter_container
```

Default container for all enter commands: `mdmitry1/python311-dev:latest`

### Run mdmitry1/python311-dev container in Ubuntu 24.04 using X11 forwarding

1. Install `socat`

```bash
sudo apt install socat
```

2. Run command

```bash
enter_container_x11_forwarding
```

### Run mdmitry1/python311-dev container in Windows11/WSL2 with wslg installed

Run command

```
enter_container_wslg
```

