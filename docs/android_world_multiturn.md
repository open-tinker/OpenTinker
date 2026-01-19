# LLM Game Agent (AndroidWorld Multi-Turn)

This example demonstrates training a language model to complete tasks in the Android operating system environment using AndroidWorld.

## Overview

**AndroidWorld** is a dynamic benchmarking environment for autonomous agents to interact with the Android operating system. The agent perceives the screen via a list of UI elements and interacts by performing actions like clicking, typing, and scrolling.

Tasks include:
- Adding contacts
- Managing settings
- Browsing information
- Sending messages
- And more...

## Prerequisites

1.  Complete the [Installation](../README.md#-installation) steps.
2.  **Environment Setup**: You must install the Android SDK and run an Emulator. See the **[Detailed Environment Setup](#detailed-environment-setup)** section below for instructions.
3.  Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the AndroidWorld Environment (Server Side)

Before starting the environment server, ensure your Android Emulator is running (see setup below).

```bash
python -m opentinker.environment.android_world.android_world_server \
    --port 8092 \
    --max_steps 50 \
    --split train
```

**Server Options:**

- `--port`: Server port (default: 8082, recommend 8092 to match client config)
- `--max_steps`: Max steps per episode (default: 50)
- `--split`: Dataset split (`train`, `eval_in_distribution`, `eval_out_of_distribution`)
- `--shards`: Number of parallel server instances (for parallel training)

## Step 3: Run Training

```bash
python opentinker/client/android_world_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=4 \
    val_batch_size=50 \
    num_steps=1000 \
    save_freq=20000 \
    test_freq=10 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=8092 \
    interaction.config.env_host=<env_server_endpoint>
```

**Training Parameters:**

- `num_steps`: Total training steps (alternative: use `num_epochs`)
- `batch_size`: Training batch size
- `val_batch_size`: Validation samples per evaluation
- `test_freq`: Validation frequency (every N steps)
- `adv_estimator`: Advantage estimator (`gae`, `grpo`, `grpo_per_step`)

## Reward Structure

| Event            | Reward |
| :--------------- | ------ |
| Task Success     | +10.0  |
| Task Failure     | -1.0   |
| Per Step Penalty | -0.01  |
| Invalid Action   | -0.1   |

## Example Actions

The agent interacts with the environment by outputting JSON commands referencing UI element indices:

- **Click**: `{"action_type": "click", "index": 4}`
- **Type**: `{"action_type": "input_text", "text": "Alice", "index": 2}`
- **Scroll**: `{"action_type": "scroll", "direction": "down"}`
- **Open App**: `{"action_type": "open_app", "app_name": "Settings"}`
- **Navigate Home**: `{"action_type": "navigate_home"}`
- **Navigate Back**: `{"action_type": "navigate_back"}`
- **Answer Question**: `{"action_type": "answer", "text": "It is 5 PM."}`
- **Finish Task**: `{"action_type": "status", "goal_status": "complete"}`

## Configuration Reference

See [`opentinker/client/client_config/android_world_param.yaml`](../opentinker/client/client_config/android_world_param.yaml) for full configuration options.

---

## Detailed Environment Setup

### 1. Android SDK & Command Line Tools

If you do not have Android Studio installed, you can set up the command-line tools manually.

1.  **Create Directory Structure:**
    ```bash
    mkdir -p /usr/local/android-sdk/cmdline-tools
    cd /usr/local/android-sdk/cmdline-tools
    ```

2.  **Download Command Line Tools:**
    ```bash
    wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O cmdline-tools.zip
    unzip cmdline-tools.zip
    mv cmdline-tools latest
    rm cmdline-tools.zip
    ```

3.  **Install SDK Components:**
    ```bash
    export ANDROID_HOME=/usr/local/android-sdk
    export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH

    # Accept licenses
    yes | sdkmanager --licenses --sdk_root=$ANDROID_HOME

    # Install Platform Tools (adb), Android 33 Platform, and Build Tools
    sdkmanager "platform-tools" "platforms;android-33" "build-tools;34.0.0" "emulator" --sdk_root=$ANDROID_HOME
    ```

4.  **Configure Environment Variables:**
    Add the following to your shell configuration file (`~/.bashrc` or `~/.zshrc`):
    ```bash
    export JAVA_HOME="/usr/local/android-studio/jbr" # Or your JDK path
    export ANDROID_HOME="/usr/local/android-sdk"
    export PATH="$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"
    ```

### 2. Create Android Virtual Device (AVD)

Create an AVD named `AndroidWorldAvd` targeting Android 13 (Tiramisu, API 33).

1.  **Install System Image:**
    *   For x86_64 (Standard PC):
        ```bash
        sdkmanager "system-images;android-33;google_apis;x86_64" --sdk_root=$ANDROID_HOME
        ```
    *   For ARM64 (Apple Silicon or Software Emulation on x86):
        ```bash
        sdkmanager "system-images;android-33;google_apis;arm64-v8a" --sdk_root=$ANDROID_HOME
        ```

2.  **Create AVD:**
    ```bash
    echo "no" | avdmanager create avd --name AndroidWorldAvd --package "system-images;android-33;google_apis;x86_64" --device "pixel_6"
    ```
    *(Replace `x86_64` with `arm64-v8a` if applicable)*

### 3. Launch Emulator

Start the emulator in a separate terminal or background process using the `sg` command to ensure correct group permissions (e.g., `kvm`).

*   **Standard Launch (with GUI):**
    ```bash
    sg kvm -c "emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554"
    ```

*   **Headless Launch (Server/Docker):**
    ```bash
    sg kvm -c "emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 -no-window -no-audio"
    ```

*   **Software Emulation (No KVM):**
    If hardware acceleration is unavailable, add `-accel off`. **Warning: Performance will be very low.**
    ```bash
    emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 -no-window -no-audio -accel off
    ```

## Troubleshooting

*   **"KVM is not found"**: Ensure virtualization is enabled in your BIOS/Hypervisor. On Linux, check permissions for `/dev/kvm`. If in a container, run with `--device /dev/kvm`.
*   **Emulator crashes immediately**: Check logs. If running x86_64 image on ARM or vice-versa, the emulator will fail. Use the correct system image for your host architecture.
*   **"ADB command not found"**: Ensure `platform-tools` is in your `$PATH`.
*   **"Process system isn't responding"**: Common in software emulation (`-accel off`). Wait for the system to stabilize or dismiss the dialog.