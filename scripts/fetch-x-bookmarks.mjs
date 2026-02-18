#!/usr/bin/env node

/**
 * Script to fetch X bookmarks and add them to news.ts
 * 
 * This script:
 * - Fetches your latest X bookmarks using the X API v2
 * - Formats them as news items with dates and links
 * - Adds them to the top of news.ts
 * - Automatically skips duplicates (checks existing bookmark IDs)
 * 
 * Usage:
 *   X_BEARER_TOKEN=your_token X_USERNAME=your_username npm run fetch-x-bookmarks
 * 
 * Environment Variables:
 *   X_BEARER_TOKEN (required) - Your X API Bearer Token
 *   X_USERNAME (optional) - Your X username (defaults to 'gauri__gupta')
 *   MAX_BOOKMARKS (optional) - Max bookmarks to fetch (defaults to 5)
 * 
 * Automation:
 *   This script runs automatically daily via GitHub Actions (.github/workflows/fetch-x-bookmarks.yml)
 *   To set up automation, add X_BEARER_TOKEN as a GitHub secret in your repository settings.
 */

import { readFileSync, writeFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Get environment variables
const accessToken = process.env.X_ACCESS_TOKEN || process.env.X_BEARER_TOKEN;
const username = process.env.X_USERNAME || "gauri__gupta";
const maxResults = parseInt(process.env.MAX_BOOKMARKS || "5", 10);

if (!accessToken) {
  console.error(
    "Error: X_ACCESS_TOKEN (or X_BEARER_TOKEN) environment variable is required"
  );
  console.error(
    "\nFor bookmarks, this must be an OAuth2 *user* access token (Authorization Code with PKCE), not an app-only token."
  );
  console.error("\nRequired scopes:");
  console.error("  - tweet.read");
  console.error("  - users.read");
  console.error("  - bookmark.read");
  console.error("\nUsage:");
  console.error(
    "  X_ACCESS_TOKEN=your_user_token X_USERNAME=your_username npm run fetch-x-bookmarks"
  );
  process.exit(1);
}

/**
 * Get user ID from username
 */
async function getUserIdFromUsername(accessToken, username) {
  const url = `https://api.x.com/2/users/by/username/${username}?user.fields=id`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(
      `Failed to get user ID: ${response.status} ${JSON.stringify(error)}`
    );
  }

  const data = await response.json();
  return data.data.id;
}

/**
 * Fetch X bookmarks for a user
 */
async function fetchXBookmarks(accessToken, userId, maxResults = 10) {
  const url = new URL(`https://api.x.com/2/users/${userId}/bookmarks`);
  url.searchParams.append("max_results", maxResults.toString());
  url.searchParams.append(
    "tweet.fields",
    "created_at,author_id,public_metrics,text"
  );
  url.searchParams.append("user.fields", "username,name");
  url.searchParams.append("expansions", "author_id");

  const response = await fetch(url.toString(), {
    method: "GET",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    const hint =
      response.status === 401 || response.status === 403
        ? "\nHint: The bookmarks endpoint requires an OAuth2 *user* access token with scopes tweet.read, users.read, bookmark.read. App-only bearer tokens will fail."
        : "";
    throw new Error(
      `Failed to fetch X bookmarks: ${response.status} ${JSON.stringify(error)}${hint}`
    );
  }

  const data = await response.json();

  // Map author information to tweets
  if (data.includes?.users) {
    const userMap = new Map(
      data.includes.users.map((user) => [user.id, user])
    );
    data.data = data.data.map((tweet) => ({
      ...tweet,
      author_username: userMap.get(tweet.author_id)?.username,
      author_name: userMap.get(tweet.author_id)?.name,
      url: `https://x.com/${userMap.get(tweet.author_id)?.username}/status/${tweet.id}`,
    }));
  }

  return data;
}

/**
 * Format a bookmark as a news item
 */
function formatBookmarkAsNews(bookmark) {
  const date = new Date(bookmark.created_at);
  const formattedDate = date.toLocaleDateString("en-US", {
    month: "long",
    year: "numeric",
  });

  // Clean up the text and create a link
  const text = bookmark.text
    .replace(/\n/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  const author = bookmark.author_username || bookmark.author_name || "X";
  const tweetUrl = bookmark.url || `https://x.com/${author}/status/${bookmark.id}`;

  const content = `Bookmarked from <a href='${tweetUrl}' target='_blank' rel='noreferrer'>@${author}</a>: ${text}`;

  return {
    date: formattedDate,
    content,
  };
}

async function main() {
  try {
    console.log(`Fetching X bookmarks for @${username}...`);

    // Get user ID
    const userId = await getUserIdFromUsername(accessToken, username);
    console.log(`Found user ID: ${userId}`);

    // Fetch bookmarks
    const response = await fetchXBookmarks(accessToken, userId, maxResults);
    console.log(`Fetched ${response.data.length} bookmarks`);

    if (response.data.length === 0) {
      console.log("No bookmarks found.");
      return;
    }

    // Read existing news.ts file
    const newsFilePath = join(__dirname, "../src/data/news.ts");
    const newsFileContent = readFileSync(newsFilePath, "utf-8");

    // Parse existing news items (simple regex-based parsing)
    // Extract the news array content
    const arrayMatch = newsFileContent.match(/export const news = \[([\s\S]*?)\];/);
    if (!arrayMatch) {
      throw new Error("Could not parse news.ts file");
    }

    let existingNewsContent = arrayMatch[1].trim();
    
    // Remove trailing comma if present
    existingNewsContent = existingNewsContent.replace(/,\s*$/, "");

    // Extract existing bookmark IDs from news.ts to prevent duplicates
    // Look for URLs that match the pattern: https://x.com/.../status/{id}
    const existingBookmarkIds = new Set();
    const bookmarkIdRegex = /https:\/\/x\.com\/[^\/]+\/status\/(\d+)/g;
    let match;
    while ((match = bookmarkIdRegex.exec(newsFileContent)) !== null) {
      existingBookmarkIds.add(match[1]);
    }

    // Format bookmarks as news items and filter out duplicates
    const allNewNewsItems = response.data.map(formatBookmarkAsNews);
    const newNewsItems = allNewNewsItems.filter((item, index) => {
      const bookmarkId = response.data[index].id;
      if (existingBookmarkIds.has(bookmarkId)) {
        console.log(`  ⏭️  Skipping duplicate bookmark: ${bookmarkId}`);
        return false;
      }
      return true;
    });

    if (newNewsItems.length === 0) {
      console.log("\n✅ No new bookmarks to add (all are already in news.ts)");
      return;
    }

    // Create new news items as TypeScript objects
    const newNewsItemsCode = newNewsItems
      .map(
        (item) => `  {
    date: "${item.date}",
    content: ${JSON.stringify(item.content)},
  }`
      )
      .join(",\n");

    // Combine: new items first, then existing (only if existing content exists)
    const updatedNewsContent = existingNewsContent
      ? `export const news = [
${newNewsItemsCode},
${existingNewsContent}
];

export default news;`
      : `export const news = [
${newNewsItemsCode}
];

export default news;`;

    // Write back to file
    writeFileSync(newsFilePath, updatedNewsContent, "utf-8");

    console.log(`\n✅ Successfully added ${newNewsItems.length} new bookmark(s) to news.ts`);
    if (allNewNewsItems.length > newNewsItems.length) {
      console.log(`   (Skipped ${allNewNewsItems.length - newNewsItems.length} duplicate(s))`);
    }
    console.log("\nNew items added:");
    newNewsItems.forEach((item, idx) => {
      console.log(`  ${idx + 1}. ${item.date}: ${item.content.substring(0, 60)}...`);
    });
  } catch (error) {
    console.error("Error:", error.message);
    if (error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();

