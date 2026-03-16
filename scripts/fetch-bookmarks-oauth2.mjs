#!/usr/bin/env node

/**
 * Fetches X bookmarks using Playwright (real Chromium browser).
 * Bypasses Cloudflare bot detection — works reliably with session cookies.
 */

import { config } from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { chromium } from 'playwright';

config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const X_COOKIES = process.env.X_COOKIES;
const X_USERNAME = process.env.X_USERNAME || 'Abhiram2k03';
const MAX_BOOKMARKS = parseInt(process.env.MAX_BOOKMARKS || '10');
const DEBUG = process.env.DEBUG === 'true';

if (!X_COOKIES) {
  console.error('❌ Error: X_COOKIES is not set in .env');
  console.error('\nHow to get it:');
  console.error('  1. Open x.com in browser (logged in)');
  console.error('  2. DevTools → Network → click any request to x.com');
  console.error('  3. Request Headers → copy the full "Cookie" value');
  console.error("  4. In .env: X_COOKIES='<paste here>'");
  process.exit(1);
}

// Parse flat cookie string into Playwright cookie objects
function parseCookies(cookieStr) {
  const cookies = [];
  for (const part of cookieStr.split(';')) {
    const eqIdx = part.indexOf('=');
    if (eqIdx === -1) continue;
    const name = part.slice(0, eqIdx).trim();
    const value = part.slice(eqIdx + 1).trim();
    if (!name) continue;
    // Set for both x.com and twitter.com
    for (const domain of ['.x.com', '.twitter.com']) {
      cookies.push({ name, value, domain, path: '/', secure: true, sameSite: 'None' });
    }
  }
  return cookies;
}

async function main() {
  let browser;
  try {
    console.log('🚀 Fetching X bookmarks...');
    console.log(`📝 Username: ${X_USERNAME}`);
    console.log(`📊 Max bookmarks: ${MAX_BOOKMARKS}`);

    browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
      userAgent:
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      viewport: { width: 1280, height: 900 },
    });

    // Restore session
    console.log('🍪 Restoring browser session...');
    await context.addCookies(parseCookies(X_COOKIES));

    const page = await context.newPage();

    if (DEBUG) page.on('console', (msg) => console.log('  [browser]', msg.text()));

    console.log('🌐 Navigating to bookmarks...');
    await page.goto('https://x.com/i/bookmarks', { waitUntil: 'load', timeout: 30000 });

    // Check we're actually logged in (not redirected to login page)
    const url = page.url();
    if (url.includes('/login') || url.includes('/i/flow/login')) {
      throw new Error('Redirected to login page — cookies are expired. Re-copy from DevTools.');
    }

    // Wait for tweets to appear
    await page.waitForSelector('article[data-testid="tweet"]', { timeout: 15000 });
    console.log('✅ Bookmarks page loaded');

    const bookmarks = [];
    const seenIds = new Set();

    // Scroll and collect until we have enough
    let noNewCount = 0;
    while (bookmarks.length < MAX_BOOKMARKS && noNewCount < 3) {
      const tweets = await page.$$eval('article[data-testid="tweet"]', (articles) => {
        return articles.map((article) => {
          // Tweet URL and ID
          const statusLink = article.querySelector('a[href*="/status/"]');
          const href = statusLink?.getAttribute('href') ?? '';
          const idMatch = href.match(/\/status\/(\d+)/);
          const id = idMatch?.[1] ?? null;

          // Author username from the status link (/@username/status/...)
          const usernameMatch = href.match(/^\/([^/]+)\/status\//);
          const username = usernameMatch?.[1] ?? null;

          // Tweet text
          const textEl = article.querySelector('[data-testid="tweetText"]');
          const text = textEl?.innerText ?? '';

          // Timestamp
          const timeEl = article.querySelector('time');
          const datetime = timeEl?.getAttribute('datetime') ?? null;

          return { id, username, text, datetime };
        }).filter((t) => t.id !== null);
      });

      let added = 0;
      for (const tweet of tweets) {
        if (!seenIds.has(tweet.id)) {
          seenIds.add(tweet.id);
          bookmarks.push(tweet);
          added++;
          if (DEBUG) console.log(`  Got: ${tweet.id} — ${tweet.text.substring(0, 60)}`);
        }
      }

      if (added === 0) {
        noNewCount++;
      } else {
        noNewCount = 0;
      }

      if (bookmarks.length >= MAX_BOOKMARKS) break;

      // Scroll down to load more
      await page.evaluate(() => window.scrollBy(0, window.innerHeight * 2));
      await page.waitForTimeout(1500);
    }

    await browser.close();
    browser = null;

    const collected = bookmarks.slice(0, MAX_BOOKMARKS);
    console.log(`✅ Fetched ${collected.length} bookmarks`);

    if (collected.length === 0) {
      console.log('ℹ️  No bookmarks found');
      return;
    }

    // Load existing news to skip duplicates
    const existingPosts = new Set();
    const newsDataPath = join(__dirname, '..', 'src', 'data', 'news.ts');

    if (existsSync(newsDataPath)) {
      const newsContent = readFileSync(newsDataPath, 'utf-8');
      for (const match of newsContent.matchAll(/\/status\/(\d+)/g)) {
        existingPosts.add(match[1]);
      }
    }

    const newNews = [];
    console.log('\n📝 Processing bookmarks for News...');

    for (const bookmark of collected) {
      if (existingPosts.has(bookmark.id)) {
        console.log(`⏭️  Skipping existing: ${bookmark.id}`);
        continue;
      }

      const date = bookmark.datetime ? new Date(bookmark.datetime) : new Date();
      const monthYear = date.toLocaleDateString('en-US', { month: 'long', year: 'numeric' });

      const tweetUrl = `https://x.com/${bookmark.username}/status/${bookmark.id}`;
      let text = bookmark.text.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim();
      text = text.replace(/https:\/\/t\.co\/\w+\s*$/, '').trim();

      const content = `<a href='${tweetUrl}' target='_blank' rel='noreferrer'>${text}</a>`;
      newNews.push({ date: monthYear, content });
      console.log(`✅ Processed: ${tweetUrl}`);
    }

    if (newNews.length === 0) {
      console.log('ℹ️  No new bookmarks to add');
      return;
    }

    console.log(`\n📝 Updating news.ts with ${newNews.length} new items...`);

    let newsContent = existsSync(newsDataPath)
      ? readFileSync(newsDataPath, 'utf-8')
      : `export const news = [];\n\nexport default news;\n`;

    const newNewsObjects = newNews
      .map((item) => {
        const safeContent = item.content.replace(/"/g, '\\"');
        return `  {\n    date: "${item.date}",\n    content: "${safeContent}",\n  }`;
      })
      .join(',\n');

    const arrayMatch = newsContent.match(/export const news = \[([\s\S]*?)\];/);
    if (arrayMatch) {
      const existingNewsStr = arrayMatch[1].trim();
      const cleanExisting = existingNewsStr.replace(/,$/, '');
      const updatedNews = newNewsObjects + (cleanExisting ? `,\n${cleanExisting}` : '');
      const newContent = newsContent.replace(
        /export const news = \[([\s\S]*?)\];/,
        `export const news = [\n${updatedNews}\n];`
      );
      writeFileSync(newsDataPath, newContent, 'utf-8');
      console.log('✅ Updated news.ts');
    }

    console.log(`\n🎉 Successfully added ${newNews.length} new X bookmarks to News!`);
  } catch (error) {
    if (browser) await browser.close();
    console.error('\n❌ ERROR OCCURRED');
    console.error('═══════════════════════════════════════');
    console.error('Error Message:', error.message);
    if (DEBUG) console.error('Error Stack:', error.stack);
    console.error('═══════════════════════════════════════');
    process.exit(1);
  }
}

main();
