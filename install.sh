#!/usr/bin/env bash
# hyprgaze installer.
#
# Interactive by default; skip prompts with --yes or --no (answers all
# prompts that way). Can be re-run safely — every step checks before
# touching user files.
#
# Steps, in order:
#   1. Check tool prerequisites
#   2. Build the venv (uv sync)
#   3. Migrate legacy ~/.config/gazefocus config / service
#   4. Install systemd --user service (~/.config/systemd/user/hyprgaze.service)
#   5. Optionally symlink bin/* into ~/.local/bin     (--with-path)
#   6. Optionally install the waybar widget           (prompt)
#   7. Optionally install the Hyprland keybind        (prompt)
#   8. Optionally run calibration now                 (prompt)
#   9. Optionally enable + start the systemd service  (prompt)
#  10. Print a usage summary
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_PATH=0
ANSWER_ALL=""                       # "y" if --yes, "n" if --no, empty = prompt
for arg in "$@"; do
    case "$arg" in
        --with-path) WITH_PATH=1 ;;
        --yes|-y)    ANSWER_ALL=y ;;
        --no)        ANSWER_ALL=n ;;
        -h|--help)
            cat <<EOH
usage: $0 [--with-path] [--yes | --no]
  --with-path   also symlink bin/hyprgaze-* into \$HOME/.local/bin
  --yes         accept all prompts (waybar / keybind / calibrate / service)
  --no          decline all prompts (install only the systemd unit)
EOH
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

ask_yn() {
    # ask_yn "prompt" → 0 yes, 1 no. Respects --yes / --no, defaults to yes.
    local prompt="$1"
    if [ -n "$ANSWER_ALL" ]; then
        [ "$ANSWER_ALL" = "y" ] && { note "$prompt → yes (--yes)"; return 0; } \
                                || { note "$prompt → no  (--no)";  return 1; }
    fi
    if [ ! -t 0 ]; then
        note "$prompt → yes (non-interactive, default)"
        return 0
    fi
    local reply
    while :; do
        printf '    %s [Y/n] ' "$prompt"
        read -r reply
        case "${reply,,}" in
            ""|y|yes) return 0 ;;
            n|no)     return 1 ;;
        esac
    done
}

# --- 1. prerequisites ---
command -v uv         >/dev/null || die "uv not found (https://github.com/astral-sh/uv)"
command -v systemctl  >/dev/null || die "systemctl not found (this installer targets systemd/Linux)"
command -v hyprctl    >/dev/null || warn "hyprctl not on PATH — hyprgaze needs Hyprland at runtime"
command -v python3    >/dev/null || die "python3 not found (used for waybar config edits)"

# --- 2. venv ---
say "syncing venv at $REPO"
( cd "$REPO" && uv sync --quiet )
HYPRGAZE_BIN="$REPO/.venv/bin/hyprgaze"
[ -x "$HYPRGAZE_BIN" ] || die "expected venv binary at $HYPRGAZE_BIN after uv sync"

# --- 3. legacy migration ---
if [ -d "$HOME/.config/gazefocus" ] && [ ! -e "$HOME/.config/hyprgaze" ]; then
    say "migrating ~/.config/gazefocus → ~/.config/hyprgaze"
    mv "$HOME/.config/gazefocus" "$HOME/.config/hyprgaze"
fi
if [ -e "$HOME/.config/systemd/user/gazefocus.service" ]; then
    say "removing legacy gazefocus.service"
    systemctl --user disable --now gazefocus.service 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/gazefocus.service"
fi

# --- 4. systemd user service ---
UNIT_SRC="$REPO/systemd/hyprgaze.service.template"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT_DST="$UNIT_DIR/hyprgaze.service"
[ -f "$UNIT_SRC" ] || die "missing template: $UNIT_SRC"
say "generating $UNIT_DST"
mkdir -p "$UNIT_DIR"
sed "s|__REPO__|$REPO|g" "$UNIT_SRC" > "$UNIT_DST"
systemctl --user daemon-reload

# --- 5. optional ~/.local/bin symlinks ---
if [ "$WITH_PATH" -eq 1 ]; then
    say "symlinking bin/hyprgaze-* → ~/.local/bin"
    mkdir -p "$HOME/.local/bin"
    for f in "$REPO"/bin/hyprgaze-*; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$HOME/.local/bin/$(basename "$f")"
    done
fi

chmod +x "$REPO"/bin/hyprgaze-* "$REPO/install.sh"

# --- 6. waybar widget ---
WAYBAR_CFG="$HOME/.config/waybar/config.jsonc"
if [ ! -f "$WAYBAR_CFG" ]; then
    WAYBAR_CFG="$HOME/.config/waybar/config"
fi
WAYBAR_STYLE="$HOME/.config/waybar/style.css"
WAYBAR_MOD="$HOME/.config/waybar/hyprgaze-module.jsonc"

if [ -f "$WAYBAR_CFG" ] && ask_yn "Install waybar widget (tray eye icon; left-click toggle, right-click recalibrate)?"; then
    say "writing $WAYBAR_MOD"
    cat > "$WAYBAR_MOD" <<EOF
{
  "custom/hyprgaze": {
    "format": "{}",
    "exec": "$REPO/bin/hyprgaze-status",
    "return-type": "json",
    "interval": 1,
    "exec-on-event": true,
    "on-click":       "$REPO/bin/hyprgaze-toggle",
    "on-click-right": "$REPO/bin/hyprgaze-recalibrate",
    "tooltip": true
  }
}
EOF

    # Inject the module into modules-right and include[] — idempotent.
    say "patching $WAYBAR_CFG"
    python3 - "$WAYBAR_CFG" "$WAYBAR_MOD" <<'PY'
import re, sys, json, io
cfg_path, mod_path = sys.argv[1], sys.argv[2]
raw = open(cfg_path).read()
stripped = re.sub(r'//.*', '', raw)
data = json.loads(stripped)
changed = False

# modules-right
mr = data.setdefault("modules-right", [])
if "custom/hyprgaze" not in mr:
    # Put just before any existing "custom/hyprwhspr", else at the front.
    try:
        i = mr.index("custom/hyprwhspr")
    except ValueError:
        i = 0
    mr.insert(i, "custom/hyprgaze")
    changed = True

# include
inc = data.setdefault("include", [])
if mod_path not in inc:
    inc.append(mod_path)
    changed = True

if changed:
    # Write back; prefer overwriting comments cleanly with pretty JSON.
    # (Backup the original first.)
    import shutil, time
    shutil.copy(cfg_path, f"{cfg_path}.bak-{int(time.time())}")
    with open(cfg_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("patched waybar config (backup made)")
else:
    print("waybar config already has custom/hyprgaze")
PY

    # CSS block — append if not already present.
    if [ -f "$WAYBAR_STYLE" ] && ! grep -q "#custom-hyprgaze" "$WAYBAR_STYLE"; then
        say "appending styles to $WAYBAR_STYLE"
        cat >> "$WAYBAR_STYLE" <<'EOF'

/* hyprgaze — matches Runic right-module convention */
#custom-hyprgaze {
  background-color: @runic-turquoise;
  color: @runic-black;
  padding: 0 10px;
  font-weight: bold;
}

#custom-hyprgaze.inactive {
  background-color: alpha(@runic-turquoise, 0.5);
}
EOF
    fi
    pkill -SIGUSR2 waybar 2>/dev/null && note "waybar reloaded (SIGUSR2)" || note "waybar not running; start it to see the widget"
fi

# --- 7. Hyprland keybind ---
HYPR_BINDS="$HOME/.config/hypr/bindings.conf"
[ -f "$HYPR_BINDS" ] || HYPR_BINDS="$HOME/.config/hypr/hyprland.conf"

if [ -f "$HYPR_BINDS" ] && ask_yn "Install Hyprland keybinds (Super+Alt+F = toggle, Super+Alt+Z = re-zero)?"; then
    appended=0
    if ! grep -q "hyprgaze-toggle" "$HYPR_BINDS"; then
        say "appending toggle keybind to $HYPR_BINDS"
        cat >> "$HYPR_BINDS" <<EOF

# hyprgaze — toggle gaze-driven window focus daemon
unbind = SUPER ALT, F  # replace any existing binding
bindd = SUPER ALT, F, Toggle hyprgaze, exec, $REPO/bin/hyprgaze-toggle
EOF
        appended=1
    else
        note "toggle keybind already present in $HYPR_BINDS"
    fi
    if ! grep -q "hyprgaze-zero" "$HYPR_BINDS"; then
        say "appending re-zero keybind to $HYPR_BINDS"
        cat >> "$HYPR_BINDS" <<EOF

# hyprgaze — quick head-pose re-baseline (stare at screen center, ~3 s)
unbind = SUPER ALT, Z  # replace any existing binding
bindd = SUPER ALT, Z, Re-zero hyprgaze, exec, $REPO/bin/hyprgaze-zero
EOF
        appended=1
    else
        note "re-zero keybind already present in $HYPR_BINDS"
    fi
    if [ "$appended" -eq 1 ]; then
        hyprctl reload >/dev/null 2>&1 && note "hyprctl reload ok" || warn "could not hyprctl reload (start Hyprland to apply)"
    fi
fi

# --- 8. calibration ---
if [ ! -f "$HOME/.config/hyprgaze/calibration.json" ]; then
    if ask_yn "Run calibration now? (~30 s, requires your eyes + camera)"; then
        "$HYPRGAZE_BIN" calibrate || warn "calibration failed or was aborted — run \`$HYPRGAZE_BIN calibrate\` later"
    else
        note "no calibration saved; daemon will fall back to linear default mapping until you run \`$HYPRGAZE_BIN calibrate\`"
    fi
else
    note "calibration already at ~/.config/hyprgaze/calibration.json (not re-running)"
fi

# --- 9. enable service ---
if ! systemctl --user is-enabled --quiet hyprgaze.service 2>/dev/null; then
    if ask_yn "Enable hyprgaze.service to start at login and start it now?"; then
        systemctl --user enable --now hyprgaze.service
        note "service active — journalctl --user -u hyprgaze -f to watch"
    fi
else
    note "hyprgaze.service already enabled"
fi

# --- 10. usage summary ---
cat <<EOF

────────────────────────────────────────
hyprgaze installed at $REPO
────────────────────────────────────────

Daily use
  • Left-click the 👁 waybar widget to pause / resume.
  • Right-click the widget to run a full recalibration (~30 s).
  • Super+Alt+F  → toggle pause/resume.
  • Super+Alt+Z  → quick head-pose re-zero (~3 s).
                   Use whenever posture shifts and gaze feels off.

Service control
  Stop:     systemctl --user stop hyprgaze
  Start:    systemctl --user start hyprgaze
  Status:   systemctl --user status hyprgaze
  Logs:     journalctl --user -u hyprgaze -f
  Disable:  systemctl --user disable --now hyprgaze

Change the keybinds
  Edit ~/.config/hypr/bindings.conf (or whichever of your hypr files has
  the bindd lines pointing at $REPO/bin/hyprgaze-*). Examples:

        unbind = SUPER ALT, F
        bindd = SUPER CTRL, G, Toggle hyprgaze, exec, $REPO/bin/hyprgaze-toggle

        unbind = SUPER ALT, Z
        bindd = SUPER CTRL, Z, Re-zero hyprgaze, exec, $REPO/bin/hyprgaze-zero

  Then:   hyprctl reload

Tune behavior
  --dwell-ms N   gaze must stay on a new window this long before focus flips
                  (default 400; edit the systemd unit's ExecStart to persist)
  --debug        preview window with live yaw/pitch + dwell bar
                  (stop the service first, then run \`$HYPRGAZE_BIN run --debug\`)

Uninstall
  systemctl --user disable --now hyprgaze
  rm ~/.config/systemd/user/hyprgaze.service
  rm ~/.config/waybar/hyprgaze-module.jsonc    (if installed)
  # then remove the waybar config lines + the hypr keybind block manually
  rm -r ~/.config/hyprgaze ~/.cache/hyprgaze   (calibration + model file)

────────────────────────────────────────
EOF
