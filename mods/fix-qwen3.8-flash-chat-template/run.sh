#!/bin/bash
set -e

cp chat_template.jinja "$WORKSPACE_DIR/fixed_chat_template.jinja"

echo "=======> Qwen3.8 fixed chat template installed"
echo "=======> use --chat-template fixed_chat_template.jinja"
