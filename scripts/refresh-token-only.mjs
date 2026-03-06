#!/usr/bin/env node

/**
 * Isolated script exclusively for refreshing X OAuth 2.0 tokens.
 * Prints the new access and refresh tokens to stdout or GITHUB_OUTPUT.
 */

import { config } from 'dotenv';
import { appendFileSync, writeFileSync } from 'fs';

config();

const X_REFRESH_TOKEN = process.env.X_REFRESH_TOKEN;
const X_CLIENT_ID = process.env.X_CLIENT_ID;
const X_CLIENT_SECRET = process.env.X_CLIENT_SECRET;

if (!X_REFRESH_TOKEN || !X_CLIENT_ID || !X_CLIENT_SECRET) {
  console.error('❌ Error: Missing required environment variables: X_REFRESH_TOKEN, X_CLIENT_ID, or X_CLIENT_SECRET');
  process.exit(1);
}

async function refreshAccessToken() {
  console.log('🔄 Refreshing access token...');
  
  const params = new URLSearchParams({
    refresh_token: X_REFRESH_TOKEN,
    grant_type: 'refresh_token',
  });

  const credentials = Buffer.from(`${X_CLIENT_ID}:${X_CLIENT_SECRET}`).toString('base64');

  try {
    const response = await fetch('https://api.x.com/2/oauth2/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${credentials}`,
      },
      body: params.toString(),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Complete error response: ${errorText}`);
      throw new Error(`Failed to refresh token: ${response.status} ${errorText}`);
    }

    const data = await response.json();
    
    console.log('✅ Access token refreshed successfully');
    
    // Always write tokens to temp files for reliable inter-step passing
    writeFileSync('/tmp/x_access_token.txt', data.access_token);
    if (data.refresh_token) {
      writeFileSync('/tmp/x_refresh_token.txt', data.refresh_token);
    }

    // Write tokens to GitHub Actions output if available
    if (process.env.GITHUB_OUTPUT) {
      console.log('📝 Writing tokens to GitHub Output for secret rotation...');
      appendFileSync(process.env.GITHUB_OUTPUT, `access_token=${data.access_token}\n`);
      if (data.refresh_token) {
        appendFileSync(process.env.GITHUB_OUTPUT, `refresh_token=${data.refresh_token}\n`);
      }
    } else {
      // Local development output
      console.log('\n⚠️ Update your .env file with the new tokens:');
      console.log(`X_ACCESS_TOKEN=${data.access_token}`);
      if (data.refresh_token) {
        console.log(`X_REFRESH_TOKEN=${data.refresh_token}`);
      }
    }

    return true;
  } catch (error) {
    console.error('❌ Failed to refresh access token:', error.message);
    process.exit(1);
  }
}

refreshAccessToken();
