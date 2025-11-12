#!/bin/bash
# SSH Setup Script for Spark Cluster
# Generates and configures SSH keys for passwordless access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/dgx_key}"
SSH_KEY_TYPE="${SSH_KEY_TYPE:-ed25519}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# Create SSH directory if it doesn't exist
setup_ssh_directory() {
    log_info "Setting up SSH directory..."
    mkdir -p "$(dirname "$SSH_KEY_PATH")"
    chmod 700 "$(dirname "$SSH_KEY_PATH")"
}

# Generate SSH key pair
generate_ssh_key() {
    if [ -f "$SSH_KEY_PATH" ]; then
        log_warn "SSH key already exists at $SSH_KEY_PATH"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing key"
            return 0
        fi
    fi

    log_info "Generating SSH key pair..."
    ssh-keygen -t "$SSH_KEY_TYPE" -f "$SSH_KEY_PATH" -N "" -C "spark-cluster-$(date +%Y%m%d)"
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "${SSH_KEY_PATH}.pub"
    log_info "SSH key generated successfully"
}

# Copy SSH key to remote host
copy_ssh_key() {
    local remote_host="$1"
    local remote_user="${2:-$USER}"

    log_info "Copying SSH key to ${remote_user}@${remote_host}..."

    if command -v ssh-copy-id &> /dev/null; then
        ssh-copy-id -i "$SSH_KEY_PATH" "${remote_user}@${remote_host}"
    else
        log_warn "ssh-copy-id not found, using manual method"
        cat "${SSH_KEY_PATH}.pub" | ssh "${remote_user}@${remote_host}" \
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
    fi

    log_info "SSH key copied successfully"
}

# Test SSH connection
test_ssh_connection() {
    local remote_host="$1"
    local remote_user="${2:-$USER}"

    log_info "Testing SSH connection to ${remote_user}@${remote_host}..."

    if ssh -i "$SSH_KEY_PATH" -o BatchMode=yes -o ConnectTimeout=5 "${remote_user}@${remote_host}" "echo 'SSH connection successful'" 2>/dev/null; then
        log_info "SSH connection test passed"
        return 0
    else
        log_error "SSH connection test failed"
        return 1
    fi
}

# Configure SSH client
configure_ssh_client() {
    local config_file="$HOME/.ssh/config"
    local remote_hosts=("$@")

    log_info "Configuring SSH client..."

    # Backup existing config
    if [ -f "$config_file" ]; then
        cp "$config_file" "${config_file}.backup.$(date +%Y%m%d_%H%M%S)"
    fi

    # Add Spark cluster configuration
    {
        echo ""
        echo "# Spark Cluster Configuration - Generated $(date)"
        echo "Host spark-* dgx-*"
        echo "    IdentityFile $SSH_KEY_PATH"
        echo "    User $USER"
        echo "    StrictHostKeyChecking accept-new"
        echo "    ServerAliveInterval 60"
        echo "    ServerAliveCountMax 3"
        echo "    Compression yes"
        echo ""
    } >> "$config_file"

    chmod 600 "$config_file"
    log_info "SSH client configured"
}

# Main setup function
main() {
    log_info "Starting SSH setup for Spark cluster..."

    setup_ssh_directory
    generate_ssh_key

    if [ $# -eq 0 ]; then
        log_info "No remote hosts specified. SSH key generated."
        log_info "Use: $0 <host1> [host2] ... to copy keys to remote hosts"
        log_info "Public key location: ${SSH_KEY_PATH}.pub"
        exit 0
    fi

    # Configure SSH client
    configure_ssh_client "$@"

    # Copy key to each host
    for host in "$@"; do
        copy_ssh_key "$host"
        test_ssh_connection "$host"
    done

    log_info "SSH setup completed successfully!"
    log_info "SSH key path: $SSH_KEY_PATH"
}

# Run main function
main "$@"
