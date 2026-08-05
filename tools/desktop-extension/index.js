#!/usr/bin/env node
// Entry point for the LocusAI Local Bash Desktop extension.
//
// Spawns mcp-remote (fetched via npx, using Claude Desktop's bundled
// Node environment -- no separate Node install required per MCPB's
// own docs) pointed at the persistent Docker-hosted MCP server, and
// pipes stdio straight through. This file exists specifically so
// entry_point and mcp_config agree with each other -- Desktop invokes
// this file directly, this file is what actually launches the proxy,
// not a placeholder pointing at unrelated config.
const { spawn } = require('child_process');

const child = spawn(
  'npx',
  ['-y', 'mcp-remote', 'http://localhost:1443/mcp'],
  { stdio: 'inherit', shell: process.platform === 'win32' }
);

child.on('exit', (code) => process.exit(code ?? 1));
child.on('error', (err) => {
  console.error('Failed to spawn mcp-remote:', err);
  process.exit(1);
});
