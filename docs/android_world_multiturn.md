# AndroidWorld Environment Documentation

**AndroidWorld** is a dynamic benchmarking environment for autonomous agents to interact with the Android operating system. This guide covers the installation, setup, and usage of the AndroidWorld environment within the OpenTinker framework.

## 1. Prerequisites

*   **OS:** Linux (Ubuntu 22.04+ recommended) or macOS.
*   **Hardware:**
    *   **CPU:** x86_64 architecture (recommended) or ARM64 (Apple Silicon).
    *   **RAM:** 16GB+ recommended.
    *   **Virtualization:** KVM (Linux) or HAXM (macOS) support is highly recommended for emulator performance. *Note: Software emulation is possible but significantly slower.*
*   **Software:** Python 3.11+, Java (JDK 17+), unzip, wget.

## 2. Installation & Setup

### 2.1. Android SDK & Command Line Tools

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

### 2.2. Create Android Virtual Device (AVD)

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

### 2.3. Launch Emulator

Start the emulator in a separate terminal or background process.

*   **Standard Launch (with GUI):**
    ```bash
    emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554
    ```

*   **Headless Launch (Server/Docker):**
    ```bash
    emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 -no-window -no-audio
    ```

*   **Software Emulation (No KVM):**
    If hardware acceleration is unavailable, add `-accel off`. **Warning: Performance will be very low.**
    ```bash
    emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 -no-window -no-audio -accel off
    ```

## 3. Usage with OpenTinker

OpenTinker provides a wrapper class `AndroidWorldGame` to interact with the environment.

### 3.1. Implementation Details
*   **Class:** `opentinker.environment.android_world.android_world_game.AndroidWorldGame`
*   **Base Class:** `AbstractGame`

### 3.2. Initialization
The environment automatically connects to the running emulator via ADB and gRPC.

```python
from opentinker.environment.android_world.android_world_game import AndroidWorldGame

# Initialize the game
env = AndroidWorldGame(
    max_steps=20,
    task_types=["ContactsAddContact"] # Optional: Specify task
)

# Reset to start a new episode
observation = env.reset()
print(observation)
```

### 3.3. Action Space
The agent communicates using structured text commands.

| Action | Format | Description |
| :--- | :--- | :--- |
| **Tap** | `tap(x, y)` | Taps the screen at coordinates (x, y). |
| **Type** | `type("text")` | Types the specified text string. |
| **Scroll** | `scroll(direction)` | Scrolls `up`, `down`, `left`, or `right`. |
| **Home** | `home` | Presses the Home button. |
| **Back** | `back` | Presses the Back button. |
| **Enter** | `enter` | Presses the Enter/Return key. |

### 3.4. Observation Space
The observation returned by the environment includes:
1.  **Task Description:** The goal the agent needs to achieve.
2.  **Visible Elements:** A text representation of interactable UI elements (buttons, text fields) visible on the current screen.

**Example Observation:**
```text
Task: Add a contact named "Alice" with phone "123-456-7890".

=== Current Screen ===
Visible Elements:
- Create new contact at [800, 2000]
- Search contacts at [100, 150]

=== Available Actions ===
- tap(x, y)
- type(text)
...
```

## 4. Troubleshooting

*   **"KVM is not found"**: Ensure virtualization is enabled in your BIOS/Hypervisor. On Linux, check permissions for `/dev/kvm`. If in a container, run with `--device /dev/kvm`.
*   **Emulator crashes immediately**: Check logs. If running x86_64 image on ARM or vice-versa, the emulator will fail. Use the correct system image for your host architecture.
*   **"ADB command not found"**: Ensure `platform-tools` is in your `$PATH`.
*   **"Process system isn't responding"**: Common in software emulation (`-accel off`). Wait for the system to stabilize or dismiss the dialog.
