#!/bin/bash
# Tailscale VPN Setup Script for Spark Cluster
# Installs and configures Tailscale for secure cluster communication

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../config"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        log_error "Cannot detect OS"
        exit 1
    fi
    log_debug "Detected OS: $OS $VER"
}

# Install Tailscale
install_tailscale() {
    log_info "Installing Tailscale..."

    case "$OS" in
        ubuntu|debian)
            curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.noarmor.gpg | sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
            curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.tailscale-keyring.list | sudo tee /etc/apt/sources.list.d/tailscale.list
            sudo apt-get update
            sudo apt-get install -y tailscale
            ;;
        centos|rhel|rocky)
            sudo dnf config-manager --add-repo https://pkgs.tailscale.com/stable/rhel/8/tailscale.repo
            sudo dnf install -y tailscale
            ;;
        *)
            log_error "Unsupported OS: $OS"
            log_info "Please install Tailscale manually: https://tailscale.com/download"
            exit 1
            ;;
    esac

    log_info "Tailscale installed successfully"
}

# Check if Tailscale is installed
check_tailscale_installed() {
    if command -v tailscale &> /dev/null; then
        log_info "Tailscale is already installed"
        return 0
    else
        return 1
    fi
}

# Start and enable Tailscale
enable_tailscale() {
    log_info "Enabling Tailscale service..."
    sudo systemctl enable --now tailscaled
    log_info "Tailscale service enabled"
}

# Connect to Tailscale
connect_tailscale() {
    local auth_key="$1"
    local hostname="$2"

    log_info "Connecting to Tailscale network..."

    if [ -n "$auth_key" ]; then
        sudo tailscale up --authkey="$auth_key" ${hostname:+--hostname="$hostname"} --accept-routes
    else
        log_warn "No auth key provided. Manual authentication required."
        sudo tailscale up ${hostname:+--hostname="$hostname"} --accept-routes
    fi

    log_info "Connected to Tailscale"
}

# Get Tailscale IP
get_tailscale_ip() {
    log_info "Retrieving Tailscale IP..."
    local tailscale_ip=$(tailscale ip -4)

    if [ -n "$tailscale_ip" ]; then
        log_info "Tailscale IPv4: $tailscale_ip"
        echo "$tailscale_ip"

        # Save to config
        mkdir -p "$CONFIG_DIR"
        echo "TAILSCALE_IP=$tailscale_ip" > "$CONFIG_DIR/tailscale.env"
        echo "TAILSCALE_HOSTNAME=$(hostname)" >> "$CONFIG_DIR/tailscale.env"
        echo "TAILSCALE_STATUS=$(tailscale status --json | jq -r '.Self.HostName')" >> "$CONFIG_DIR/tailscale.env"
    else
        log_error "Failed to retrieve Tailscale IP"
        return 1
    fi
}

# Display Tailscale status
show_status() {
    log_info "Tailscale Status:"
    tailscale status
    echo ""
    log_info "Network devices:"
    tailscale status --json | jq -r '.Peer[] | "\(.HostName): \(.TailscaleIPs[0])"'
}

# Configure advertise routes (for subnet router)
configure_routes() {
    local subnet="$1"

    if [ -n "$subnet" ]; then
        log_info "Configuring as subnet router for $subnet..."
        sudo tailscale up --advertise-routes="$subnet" --accept-routes

        # Enable IP forwarding
        sudo sysctl -w net.ipv4.ip_forward=1
        sudo sysctl -w net.ipv6.conf.all.forwarding=1

        # Make permanent
        echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
        echo "net.ipv6.conf.all.forwarding=1" | sudo tee -a /etc/sysctl.conf

        log_info "Subnet router configured"
    fi
}

# Main setup function
main() {
    log_info "Starting Tailscale setup for Spark cluster..."

    detect_os

    if ! check_tailscale_installed; then
        install_tailscale
    fi

    enable_tailscale

    # Parse arguments
    AUTH_KEY="${1:-}"
    HOSTNAME="${2:-spark-$(hostname)}"
    SUBNET="${3:-}"

    if [ -z "$AUTH_KEY" ]; then
        log_warn "No auth key provided. You can generate one at: https://login.tailscale.com/admin/settings/keys"
        log_info "Usage: $0 <auth-key> [hostname] [subnet-to-advertise]"
    fi

    connect_tailscale "$AUTH_KEY" "$HOSTNAME"

    if [ -n "$SUBNET" ]; then
        configure_routes "$SUBNET"
    fi

    sleep 2
    get_tailscale_ip
    show_status

    log_info "Tailscale setup completed successfully!"
    log_info "Config saved to: $CONFIG_DIR/tailscale.env"
}

# Run main function
main "$@"
