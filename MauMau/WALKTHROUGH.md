# MauMau Project - Setup Walkthrough

This guide explains how to start from a Windows environment, access the WSL (Linux) environment, and run the MauMau web project.

## Step 1: Open PowerShell
1. On your Windows desktop, press `Windows Key`.
2. Type `PowerShell`.
3. Press `Enter` to open it.

## Step 2: Enter WSL (Windows Subsystem for Linux)
In the PowerShell window, type the following command to enter your default Linux distribution:

```powershell
wsl
```

*You should now see a linux-style prompt (e.g., user@hostname $).*

## Step 3: Navigate to the Project Directory
Navigate to the `MauMau` folder within the `ai-lab` repository:

```bash
cd ~/projects/ai-lab/MauMau
```

## Step 4: Start a Local Web Server
You need a simple web server to serve the HTML file. Python is pre-installed in most Linux environments and is perfect for this.

Type the following command:

```bash
python3 -m http.server 8000
```

*You should see output indicating the server is serving HTTP on port 8000.*

## Step 5: View in Browser
1. Open your web browser on Windows (Chrome, Firefox, Edge, etc.).
2. In the address bar, type:
   `http://localhost:8000`
3. Press `Enter`.

You should now see the Tokyo Night Mau Mau cards rendered in your browser!

## Step 6: Verify Game Engine
To verify that the game logic (shuffling, validation, rules) is working correctly without Node.js, we have included a browser-based test runner.

1. Ensure your local server is running (Step 4).
2. In your browser, navigate to:
   `http://localhost:8000/test_engine.html`
3. You should see a log output ending with `SUCCESS: Basic logic seems correct.`


### Troubleshooting
- If port 8000 is taken, try another port: `python3 -m http.server 8080` (and use `localhost:8080`).
- To stop the server, go back to your terminal window and press `Ctrl+C`.
