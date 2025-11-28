import { useParams, Link } from "react-router-dom";
import { useState, useEffect } from "react";
import posts from "../data/posts";
import MarkdownRenderer from "../components/MarkdownRenderer";
import {
  ArrowLeft,
  Share2,
  Check,
  Twitter,
  Linkedin,
  Copy,
} from "../components/Icons";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../components/ui/dropdown-menu";

export default function BlogPost() {
  const { id } = useParams();
  const post = posts.find((p) => p.id === id);
  const [copied, setCopied] = useState(false);

  // Update meta tags for social sharing
  useEffect(() => {
    if (post) {
      // Update page title
      document.title = `${post.title} - Blog`;

      // Update or create meta tags
      const updateMetaTag = (property: string, content: string) => {
        let element = document.querySelector(`meta[property="${property}"]`);
        if (!element) {
          element = document.createElement("meta");
          element.setAttribute("property", property);
          document.head.appendChild(element);
        }
        element.setAttribute("content", content);
      };

      const updateNameMetaTag = (name: string, content: string) => {
        let element = document.querySelector(`meta[name="${name}"]`);
        if (!element) {
          element = document.createElement("meta");
          element.setAttribute("name", name);
          document.head.appendChild(element);
        }
        element.setAttribute("content", content);
      };

      // Open Graph tags (Facebook, LinkedIn)
      updateMetaTag("og:title", post.title);
      updateMetaTag("og:description", post.excerpt || "");
      updateMetaTag("og:url", window.location.href);
      updateMetaTag("og:type", "article");

      // Twitter Card tags
      updateNameMetaTag("twitter:card", "summary_large_image");
      updateNameMetaTag("twitter:title", post.title);
      updateNameMetaTag("twitter:description", post.excerpt || "");

      // Generic description
      updateNameMetaTag("description", post.excerpt || "");
    }

    return () => {
      document.title = "Blog";
    };
  }, [post]);

  const handleCopyLink = async () => {
    await navigator.clipboard.writeText(window.location.href);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleShare = (
    platform: "twitter" | "linkedin" | "facebook" | "whatsapp" | "reddit"
  ) => {
    const pageUrl = window.location.href;
    const url = encodeURIComponent(pageUrl);
    const title = encodeURIComponent(post?.title || "");

    const shareUrls: Record<string, string> = {
      twitter: `https://twitter.com/intent/tweet?url=${url}&text=${title}`,
      // LinkedIn with proper parameters
      linkedin: `https://www.linkedin.com/sharing/share-offsite/?url=${url}`,
      // Facebook sharer
      facebook: `https://www.facebook.com/sharer/sharer.php?u=${url}`,
      // WhatsApp
      whatsapp: `https://api.whatsapp.com/send?text=${title}%20${url}`,
      // Reddit
      reddit: `https://reddit.com/submit?url=${url}&title=${title}`,
    };

    const shareUrl = shareUrls[platform];
    if (!shareUrl) return;

    window.open(shareUrl, "_blank", "width=600,height=600");
  };

  if (!post) {
    return (
      <div className="flex items-center justify-center px-6 py-32">
        <div className="text-center space-y-4">
          <h1 className="text-xl font-semibold">Post not found</h1>
          <Link
            to="/blogs"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to blogs
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      <div className="max-w-[680px] mx-auto px-6 py-16 md:py-24">
        {/* Top Bar with Back Link and Share */}
        <div className="flex items-center justify-between mb-12 animate-slide-up">
          <Link
            to="/blogs"
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform duration-200" />
            Blogs
          </Link>

          {/* Share Button (shadcn DropdownMenu) */}
          <DropdownMenu>
            <div>
              <DropdownMenuTrigger asChild>
                <button className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-lg hover:bg-muted/50">
                  <Share2 className="w-4 h-4" />
                  <span className="hidden sm:inline">Share</span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent sideOffset={8} className="w-48">
                <DropdownMenuItem
                  onClick={handleCopyLink}
                  className="flex items-center gap-3"
                >
                  <Copy className="w-4 h-4" />
                  <span>{copied ? "Copied!" : "Copy link"}</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleShare("twitter")}
                  className="flex items-center gap-3"
                >
                  <Twitter className="w-4 h-4" />
                  <span>Share on Twitter</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleShare("linkedin")}
                  className="flex items-center gap-3"
                >
                  <Linkedin className="w-4 h-4" />
                  <span>Share on LinkedIn</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </div>
          </DropdownMenu>
        </div>

        {/* Article Header */}
        <header
          className="mb-10 animate-slide-up"
          style={{ animationDelay: "50ms" }}
        >
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight leading-tight mb-4">
            {post.title}
          </h1>
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            <span>{post.author}</span>
            <span className="text-border">·</span>
            <span className="font-mono text-xs">{post.date}</span>
          </div>
        </header>

        {/* Article Content */}
        <article
          className="mb-12 animate-slide-up"
          style={{ animationDelay: "100ms" }}
        >
          <MarkdownRenderer content={post.content} />
        </article>

        {/* Footer */}
        <footer
          className="pt-8 border-t border-border animate-slide-up"
          style={{ animationDelay: "150ms" }}
        >
          <div className="flex items-center justify-between">
            <Link
              to="/blogs"
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform duration-200" />
              All articles
            </Link>
             <DropdownMenu>
            <div>
              <DropdownMenuTrigger asChild>
                <button className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-lg hover:bg-muted/50">
                  <Share2 className="w-4 h-4" />
                  <span className="hidden sm:inline">Share</span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent sideOffset={8} className="w-48">
                <DropdownMenuItem
                  onClick={handleCopyLink}
                  className="flex items-center gap-3"
                >
                  <Copy className="w-4 h-4" />
                  <span>{copied ? "Copied!" : "Copy link"}</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleShare("twitter")}
                  className="flex items-center gap-3"
                >
                  <Twitter className="w-4 h-4" />
                  <span>Share on Twitter</span>
                </DropdownMenuItem>
                <DropdownMenuItem
                  onClick={() => handleShare("linkedin")}
                  className="flex items-center gap-3"
                >
                  <Linkedin className="w-4 h-4" />
                  <span>Share on LinkedIn</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </div>
          </DropdownMenu>
          </div>
        </footer>
        {/* Transient visual confirmation for copy action */}
        <div
          aria-live="polite"
          className={`fixed right-6 top-20 z-50 pointer-events-none transition-opacity duration-300 ${
            copied ? "opacity-100" : "opacity-0"
          }`}
        >
          <div className="bg-foreground/95 text-background rounded-md px-4 py-2 shadow-lg flex items-center gap-3 text-sm font-medium">
            <Check className="w-4 h-4 text-green-500 font-bold" />
            <span>Link copied to clipboard</span>
          </div>
        </div>
      </div>
    </div>
  );
}
