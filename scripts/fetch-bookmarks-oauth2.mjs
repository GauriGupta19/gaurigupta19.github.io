#!/usr/bin/env node

/**
 * Script to fetch X bookmarks using OAuth 2.0
 * This works with the OAuth 2.0 token from get-oauth2-token.mjs
 */

import { config } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, writeFileSync, appendFileSync, existsSync, mkdirSync } from 'fs';

// Load environment variables
config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let X_ACCESS_TOKEN = process.env.X_ACCESS_TOKEN;
const X_REFRESH_TOKEN = process.env.X_REFRESH_TOKEN;
const X_CLIENT_ID = process.env.X_CLIENT_ID;
const X_CLIENT_SECRET = process.env.X_CLIENT_SECRET;
const X_USERNAME = process.env.X_USERNAME || 'Abhiram2k03';
const MAX_BOOKMARKS = parseInt(process.env.MAX_BOOKMARKS || '10');
const DEBUG = process.env.DEBUG === 'true';

if (!X_ACCESS_TOKEN || !X_REFRESH_TOKEN) {
  console.error('❌ Error: X_ACCESS_TOKEN or X_REFRESH_TOKEN is not set');
  console.error('\nPlease run: npm run get-oauth2-token');
  console.error('Then add both tokens to your .env file');
  process.exit(1);
}

if (!X_CLIENT_ID) {
  console.error('❌ Error: X_CLIENT_ID is not set in .env file');
  process.exit(1);
}

if (!X_CLIENT_SECRET) {
  console.error('❌ Error: X_CLIENT_SECRET is not set in .env file');
  console.error('This is required for refreshing access tokens');
  process.exit(1);
}

if (DEBUG) {
  console.log('\n🔍 DEBUG MODE ENABLED');
  console.log('═══════════════════════════════════════');
  console.log('Environment Configuration:');
  console.log('  X_USERNAME:', X_USERNAME);
  console.log('  MAX_BOOKMARKS:', MAX_BOOKMARKS);
  console.log('  X_ACCESS_TOKEN:', X_ACCESS_TOKEN ? `Set (${X_ACCESS_TOKEN.length} chars)` : 'NOT SET');
  console.log('  X_REFRESH_TOKEN:', X_REFRESH_TOKEN ? `Set (${X_REFRESH_TOKEN.length} chars)` : 'NOT SET');
  console.log('  Working Directory:', process.cwd());
  console.log('═══════════════════════════════════════\n');
}

/**
 * Refresh the OAuth 2.0 access token using the refresh token
 */
async function refreshAccessToken() {
  console.log('🔄 Refreshing access token...');
  
  const params = new URLSearchParams({
    refresh_token: X_REFRESH_TOKEN,
    grant_type: 'refresh_token',
  });

  // Create Basic Auth header with client_id:client_secret
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
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`Failed to refresh token: ${response.status} ${JSON.stringify(error)}`);
    }

    const data = await response.json();
    X_ACCESS_TOKEN = data.access_token;
    
    console.log('✅ Access token refreshed successfully');
    console.log('ℹ️  New token expires in:', data.expires_in, 'seconds');
    console.log('\n⚠️  Update your .env file with the new tokens:');
    console.log(`X_ACCESS_TOKEN=${data.access_token}`);
    if (data.refresh_token) {
      console.log(`X_REFRESH_TOKEN=${data.refresh_token}`);
    }
    
    // Write tokens to GitHub Actions output if available
    if (process.env.GITHUB_OUTPUT) {
      console.log('📝 Writing tokens to GitHub Output for secret rotation...');
      appendFileSync(process.env.GITHUB_OUTPUT, `access_token=${data.access_token}\n`);
      if (data.refresh_token) {
        appendFileSync(process.env.GITHUB_OUTPUT, `refresh_token=${data.refresh_token}\n`);
      }
    }
    
    console.log('');
    
    return data.access_token;
  } catch (error) {
    console.error('❌ Failed to refresh access token:', error.message);
    console.error('\nYou may need to re-authorize by running: npm run get-oauth2-token');
    throw error;
  }
}

/**
 * Make an API request with automatic token refresh on 401 errors
 */
async function fetchWithTokenRefresh(url, options = {}) {
  const makeRequest = async (token) => {
    return fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${token}`,
      },
    });
  };

  let response = await makeRequest(X_ACCESS_TOKEN);

  // If we get a 401, try refreshing the token once
  if (response.status === 401) {
    console.log('⚠️  Access token expired, refreshing...');
    await refreshAccessToken();
    response = await makeRequest(X_ACCESS_TOKEN);
  }

  return response;
}

async function getUserIdFromUsername(username) {
  const url = `https://api.x.com/2/users/by/username/${username}?user.fields=id`;
  
  if (DEBUG) {
    console.log('\n🔍 Fetching user ID...');
    console.log('  URL:', url);
  }
  
  const response = await fetchWithTokenRefresh(url, {
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
  if (DEBUG) {
    console.log('  Response Status:', response.status, response.statusText);
  }
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    console.error('\n❌ Failed to get user ID');
    console.error('  Status:', response.status);
    console.error('  Error:', JSON.stringify(error, null, 2));
    throw new Error(`Failed to get user ID: ${response.status} ${JSON.stringify(error)}`);
  }
  
  const data = await response.json();
  if (DEBUG) {
    console.log('  User ID:', data.data.id);
  }
  return data.data.id;
}

async function fetchXBookmarks(userId, maxResults = 10) {
  const url = new URL(`https://api.x.com/2/users/${userId}/bookmarks`);
  url.searchParams.append('max_results', maxResults.toString());
  url.searchParams.append('tweet.fields', 'created_at,author_id,public_metrics,text,attachments');
  url.searchParams.append('user.fields', 'username,name');
  url.searchParams.append('expansions', 'author_id,attachments.media_keys');
  url.searchParams.append('media.fields', 'url,preview_image_url,type');
  
  if (DEBUG) {
    console.log('\n🔍 Fetching bookmarks...');
    console.log('  User ID:', userId);
    console.log('  Max Results:', maxResults);
    console.log('  URL:', url.toString());
  }
  
  const response = await fetchWithTokenRefresh(url.toString(), {
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
  if (DEBUG) {
    console.log('  Response Status:', response.status, response.statusText);
  }
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Unknown error' }));
    console.error('\n❌ Failed to fetch bookmarks');
    console.error('  Status:', response.status);
    console.error('  Error:', JSON.stringify(error, null, 2));
    throw new Error(`Failed to fetch bookmarks: ${response.status} ${JSON.stringify(error)}`);
  }
  
  const data = await response.json();
  
  if (DEBUG) {
    console.log('  Bookmarks found:', data.data?.length || 0);
    console.log('  Users included:', data.includes?.users?.length || 0);
  }
  
  // Map author information to tweets
  if (data.includes?.users) {
    const userMap = new Map(data.includes.users.map((user) => [user.id, user]));
    data.data = data.data.map((tweet) => ({
      ...tweet,
      author_username: userMap.get(tweet.author_id)?.username,
      author_name: userMap.get(tweet.author_id)?.name,
      url: `https://x.com/${userMap.get(tweet.author_id)?.username}/status/${tweet.id}`,
    }));
  }
  
  // Map media attachments to tweets
  if (data.includes?.media) {
    const mediaMap = new Map(data.includes.media.map((m) => [m.media_key, m]));
    data.data = data.data.map((tweet) => {
      const media = tweet.attachments?.media_keys?.map((key) => mediaMap.get(key)).filter(Boolean) || [];
      return {
        ...tweet,
        media,
      };
    });
  }
  
  return data;
}

function bookmarkToMarkdown(bookmark) {
  const date = new Date(bookmark.created_at);
  const formattedDate = date.toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });
  
  const author = bookmark.author_name || bookmark.author_username || 'Unknown';
  const username = bookmark.author_username || 'unknown';
  
  let text = bookmark.text.trim();
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  text = text.replace(urlRegex, '[$1]($1)');
  
  // Get title from first line, limit to 100 chars
  const title = text.split('\n')[0].substring(0, 100);

  const mediaText = bookmark.media && bookmark.media.length > 0 ? `\n\n` + bookmark.media.map(m => {
    if (m.type === 'photo') {
      return `![Image](${m.url})`;
    } else if (m.type === 'video' || m.type === 'animated_gif') {
      return `![Video Preview](${m.preview_image_url})\n*[Video URL](${bookmark.url})*`;
    }
    return '';
  }).join('\n\n') : '';

  return `## ${title}

**Originally posted by [@${username}](https://x.com/${username})** on ${formattedDate}

[View original post on X](${bookmark.url})

---

${text}${mediaText}
`;
}

async function main() {
  try {
    console.log('🚀 Fetching X bookmarks...');
    console.log(`📝 Username: ${X_USERNAME}`);
    console.log(`📊 Max bookmarks: ${MAX_BOOKMARKS}`);
    
    const userId = await getUserIdFromUsername(X_USERNAME);
    console.log(`✅ Found user ID: ${userId}`);
    
    const bookmarksData = await fetchXBookmarks(userId, MAX_BOOKMARKS);
    
    if (!bookmarksData.data || bookmarksData.data.length === 0) {
      console.log('ℹ️  No bookmarks found');
      return;
    }
    
    console.log(`✅ Fetched ${bookmarksData.data.length} bookmarks`);
    
    const postsDir = join(__dirname, '..', 'src', 'pages', 'posts');
    const publicBlogsDir = join(__dirname, '..', 'public', 'blogs');
    
    if (!existsSync(postsDir)) {
      console.log(`  Creating posts directory: ${postsDir}`);
      mkdirSync(postsDir, { recursive: true });
    }
    if (!existsSync(publicBlogsDir)) {
      console.log(`  Creating public blogs directory: ${publicBlogsDir}`);
      mkdirSync(publicBlogsDir, { recursive: true });
    }
    
    const existingPosts = new Set();
    const postsDataPath = join(__dirname, '..', 'src', 'data', 'posts.ts');
    
    if (existsSync(postsDataPath)) {
      const postsContent = readFileSync(postsDataPath, 'utf-8');
      const idMatches = postsContent.matchAll(/id: ['"]([^'"]+)['"]/g);
      for (const match of idMatches) {
        existingPosts.add(match[1]);
      }
    }
    
    const newPosts = [];
    console.log('\n📝 Processing bookmarks...');
    
    for (const bookmark of bookmarksData.data) {
      const slug = `x-bookmark-${bookmark.id}`;
      
      if (existingPosts.has(slug)) {
        console.log(`⏭️  Skipping existing post: ${slug}`);
        continue;
      }
      
      const markdown = bookmarkToMarkdown(bookmark);
      const mdPath = join(postsDir, `${slug}.md`);
      
      writeFileSync(mdPath, markdown, 'utf-8');
      console.log(`✅ Created: ${mdPath}`);
      
      const firstLine = bookmark.text.split('\n')[0].substring(0, 100);
      const excerpt = bookmark.text.substring(0, 200).trim() + (bookmark.text.length > 200 ? '...' : '');
      const date = new Date(bookmark.created_at);
      const monthYear = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });
      
      newPosts.push({
        id: slug,
        title: firstLine,
        date: monthYear,
        author: bookmark.author_name || bookmark.author_username || 'Unknown',
        categories: ['X Bookmark', 'Curated'],
        excerpt: excerpt,
      });
    }
    
    if (newPosts.length === 0) {
      console.log('ℹ️  No new posts to add');
      return;
    }
    
    console.log(`\n📝 Updating posts.ts with ${newPosts.length} new posts...`);
    
    let postsContent = existsSync(postsDataPath)
      ? readFileSync(postsDataPath, 'utf-8')
      : `const posts = [];\n\nexport { posts };\nexport default posts;\n`;
    
    const newImports = newPosts.map(post =>
      `import ${post.id.replace(/-/g, '_')}Md from '../pages/posts/${post.id}.md?raw';`
    ).join('\n');
    
    const newPostObjects = newPosts.map(post => {
      return `  {
    id: '${post.id}',
    title: ${JSON.stringify(post.title)},
    date: '${post.date}',
    author: ${JSON.stringify(post.author)},
    categories: ${JSON.stringify(post.categories)},
    excerpt: ${JSON.stringify(post.excerpt.replace(/\r?\n|\r/g, " "))},
    content: ${post.id.replace(/-/g, '_')}Md,
  }`;
    }).join(',\n');
    
    const importMatch = postsContent.match(/^(import[^;]+;?\n)*/);
    const existingImports = importMatch ? importMatch[0] : '';
    const restOfContent = postsContent.substring(existingImports.length);
    
    const arrayMatch = restOfContent.match(/const posts = \[([\s\S]*?)\];/);
    if (arrayMatch) {
      const existingPostsStr = arrayMatch[1].trim();
      const cleanExisting = existingPostsStr.replace(/,$/, '');
      const updatedPosts = cleanExisting
        ? `${cleanExisting},\n${newPostObjects}`
        : newPostObjects;
      
      const newContent = `${existingImports}${newImports}\n\n${restOfContent.replace(
        /const posts = \[([\s\S]*?)\];/,
        `const posts = [\n${updatedPosts}\n];`
      )}`;
      
      writeFileSync(postsDataPath, newContent, 'utf-8');
      console.log('✅ Updated posts.ts');
    }
    
    console.log(`\n🎉 Successfully created ${newPosts.length} new blog posts from X bookmarks!`);
  } catch (error) {
    console.error('\n❌ ERROR OCCURRED');
    console.error('═══════════════════════════════════════');
    console.error('Error Message:', error.message);
    console.error('Error Stack:', error.stack);
    console.error('═══════════════════════════════════════');
    process.exit(1);
  }
}

main();
