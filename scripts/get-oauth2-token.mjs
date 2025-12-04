#!/usr/bin/env node

/**
 * Interactive script to get OAuth 2.0 token for X Bookmarks API
 * Run this locally to authorize and get your access token
 */

import { createServer } from 'http';
import { randomBytes, createHash } from 'crypto';
import { exec } from 'child_process';
import { promisify } from 'util';
import { config } from 'dotenv';

// Load environment variables
config();

const execAsync = promisify(exec);

// Your app credentials (from X Developer Portal)
const CLIENT_ID = process.env.X_CLIENT_ID || process.argv[2];
const CLIENT_SECRET = process.env.X_CLIENT_SECRET;
const REDIRECT_URI = 'http://localhost:5173/callback';
const SCOPES = ['tweet.read', 'users.read', 'bookmark.read', 'offline.access'];

if (!CLIENT_ID || !CLIENT_SECRET) {
  console.error('❌ Error: X_CLIENT_ID and X_CLIENT_SECRET are required');
  console.error('\nUsage:');
  console.error('  node scripts/get-oauth2-token.mjs YOUR_CLIENT_ID');
  console.error('\nOr set X_CLIENT_ID and X_CLIENT_SECRET in your .env file');
  console.error('\nGet your credentials from: https://developer.x.com/en/portal/dashboard');
  process.exit(1);
}

// Generate PKCE challenge
function generateCodeChallenge() {
  const codeVerifier = randomBytes(32).toString('base64url');
  const hash = createHash('sha256').update(codeVerifier).digest('base64url');
  return { codeVerifier, codeChallenge: hash };
}

const { codeVerifier, codeChallenge } = generateCodeChallenge();
const state = randomBytes(16).toString('hex');

// Build authorization URL
const authUrl = new URL('https://twitter.com/i/oauth2/authorize');
authUrl.searchParams.set('response_type', 'code');
authUrl.searchParams.set('client_id', CLIENT_ID);
authUrl.searchParams.set('redirect_uri', REDIRECT_URI);
authUrl.searchParams.set('scope', SCOPES.join(' '));
authUrl.searchParams.set('state', state);
authUrl.searchParams.set('code_challenge', codeChallenge);
authUrl.searchParams.set('code_challenge_method', 'S256');

console.log('\n🔐 X OAuth 2.0 Authorization\n');
console.log('═══════════════════════════════════════\n');
console.log('Opening browser for authorization...\n');
console.log('If the browser doesn\'t open, visit this URL:\n');
console.log(authUrl.toString());
console.log('\n═══════════════════════════════════════\n');

// Create local server to receive callback
const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:5173`);
  
  if (url.pathname === '/callback') {
    const code = url.searchParams.get('code');
    const returnedState = url.searchParams.get('state');
    
    if (!code || returnedState !== state) {
      res.writeHead(400, { 'Content-Type': 'text/html' });
      res.end('<h1>❌ Authorization failed</h1><p>Invalid state or missing code</p>');
      server.close();
      return;
    }
    
    try {
      // Exchange code for access token
      const authHeader = 'Basic ' + Buffer.from(`${CLIENT_ID}:${CLIENT_SECRET}`).toString('base64');
      
      const tokenResponse = await fetch('https://api.twitter.com/2/oauth2/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Authorization': authHeader,
        },
        body: new URLSearchParams({
          code,
          grant_type: 'authorization_code',
          redirect_uri: REDIRECT_URI,
          code_verifier: codeVerifier,
        }),
      });
      
      const tokens = await tokenResponse.json();
      
      if (!tokenResponse.ok) {
        throw new Error(JSON.stringify(tokens, null, 2));
      }
      
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(`
        <h1>✅ Authorization Successful!</h1>
        <p>You can close this window and return to your terminal.</p>
        <style>
          body { font-family: system-ui; max-width: 600px; margin: 50px auto; padding: 20px; }
          h1 { color: #1DA1F2; }
        </style>
      `);
      
      console.log('\n✅ Authorization successful!\n');
      console.log('═══════════════════════════════════════\n');
      console.log('Add these to your .env file:\n');
      console.log(`X_OAUTH2_ACCESS_TOKEN=${tokens.access_token}`);
      console.log(`X_OAUTH2_REFRESH_TOKEN=${tokens.refresh_token}`);
      console.log('\n═══════════════════════════════════════\n');
      console.log('Token expires in:', tokens.expires_in, 'seconds');
      console.log('\nUse the refresh token to get a new access token when it expires.\n');
      
      server.close();
    } catch (error) {
      console.error('\n❌ Token exchange failed:', error.message);
      res.writeHead(500, { 'Content-Type': 'text/html' });
      res.end('<h1>❌ Token exchange failed</h1><p>Check terminal for details</p>');
      server.close();
    }
  }
});

server.listen(5173, async () => {
  console.log('Local server started on http://localhost:5173\n');
  
  // Try to open browser automatically
  const platform = process.platform;
  const command = platform === 'darwin' ? 'open' : platform === 'win32' ? 'start' : 'xdg-open';
  
  try {
    await execAsync(`${command} "${authUrl.toString()}"`);
  } catch (error) {
    console.log('Could not open browser automatically. Please open the URL manually.\n');
  }
});
