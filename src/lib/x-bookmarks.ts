/**
 * Utility functions to fetch X (Twitter) bookmarks using X API v2
 */

export interface XBookmark {
  id: string;
  text: string;
  created_at: string;
  author_id: string;
  author_username?: string;
  author_name?: string;
  url?: string;
  public_metrics?: {
    like_count: number;
    retweet_count: number;
    reply_count: number;
    quote_count: number;
  };
}

export interface XBookmarksResponse {
  data: XBookmark[];
  meta: {
    result_count: number;
    next_token?: string;
  };
}

/**
 * Fetch X bookmarks for a user
 * @param bearerToken - X API Bearer token
 * @param userId - X user ID (not username)
 * @param maxResults - Maximum number of bookmarks to fetch (default: 10, max: 100)
 * @param paginationToken - Token for pagination
 */
export async function fetchXBookmarks(
  bearerToken: string,
  userId: string,
  maxResults: number = 10,
  paginationToken?: string
): Promise<XBookmarksResponse> {
  const url = new URL(`https://api.x.com/2/users/${userId}/bookmarks`);
  url.searchParams.append("max_results", maxResults.toString());
  url.searchParams.append(
    "tweet.fields",
    "created_at,author_id,public_metrics,text"
  );
  url.searchParams.append("user.fields", "username,name");
  url.searchParams.append("expansions", "author_id");

  if (paginationToken) {
    url.searchParams.append("pagination_token", paginationToken);
  }

  const response = await fetch(url.toString(), {
    method: "GET",
    headers: {
      Authorization: `Bearer ${bearerToken}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Unknown error" }));
    throw new Error(
      `Failed to fetch X bookmarks: ${response.status} ${JSON.stringify(error)}`
    );
  }

  const data = await response.json();

  // Map author information to tweets
  if (data.includes?.users) {
    interface XUser {
      id: string;
      username?: string;
      name?: string;
    }
    const userMap = new Map(
      (data.includes.users as XUser[]).map((user) => [user.id, user])
    );
    data.data = data.data.map((tweet: XBookmark) => ({
      ...tweet,
      author_username: userMap.get(tweet.author_id)?.username,
      author_name: userMap.get(tweet.author_id)?.name,
      url: `https://x.com/${userMap.get(tweet.author_id)?.username}/status/${tweet.id}`,
    }));
  }

  return data;
}

/**
 * Get user ID from username
 * @param bearerToken - X API Bearer token
 * @param username - X username (without @)
 */
export async function getUserIdFromUsername(
  bearerToken: string,
  username: string
): Promise<string> {
  const url = new URL("https://api.x.com/2/users/by/username/" + username);
  url.searchParams.append("user.fields", "id");

  const response = await fetch(url.toString(), {
    method: "GET",
    headers: {
      Authorization: `Bearer ${bearerToken}`,
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
 * Format a bookmark as a news item
 */
export function formatBookmarkAsNews(bookmark: XBookmark): {
  date: string;
  content: string;
} {
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

