import { useParams, Link } from "react-router-dom"
import { useState, useEffect } from "react"
import posts from "../data/posts"
import MarkdownRenderer from "../components/MarkdownRenderer"
import { ArrowLeft, Share2, Check, Twitter, Linkedin, LinkIcon, Facebook, MessageCircle } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "../components/ui/dropdown-menu"

export default function BlogPost() {
  const { id } = useParams()
  const post = posts.find((p) => p.id === id)
  const [copied, setCopied] = useState(false)

  // Update meta tags for social sharing
  useEffect(() => {
    if (post) {
      // Update page title
      document.title = `${post.title} - Blog`
      
      // Update or create meta tags
      const updateMetaTag = (property: string, content: string) => {
        let element = document.querySelector(`meta[property="${property}"]`)
        if (!element) {
          element = document.createElement('meta')
          element.setAttribute('property', property)
          document.head.appendChild(element)
        }
        element.setAttribute('content', content)
      }

      const updateNameMetaTag = (name: string, content: string) => {
        let element = document.querySelector(`meta[name="${name}"]`)
        if (!element) {
          element = document.createElement('meta')
          element.setAttribute('name', name)
          document.head.appendChild(element)
        }
        element.setAttribute('content', content)
      }

      // Open Graph tags (Facebook, LinkedIn)
      updateMetaTag('og:title', post.title)
      updateMetaTag('og:description', post.excerpt || '')
      updateMetaTag('og:url', window.location.href)
      updateMetaTag('og:type', 'article')
      
      // Twitter Card tags
      updateNameMetaTag('twitter:card', 'summary_large_image')
      updateNameMetaTag('twitter:title', post.title)
      updateNameMetaTag('twitter:description', post.excerpt || '')
      
      // Generic description
      updateNameMetaTag('description', post.excerpt || '')
    }

    return () => {
      document.title = 'Blog'
    }
  }, [post])

  const handleCopyLink = async () => {
    await navigator.clipboard.writeText(window.location.href)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleShare = (platform: 'twitter' | 'linkedin' | 'facebook' | 'whatsapp' | 'reddit') => {
    const pageUrl = window.location.href
    const url = encodeURIComponent(pageUrl)
    const title = encodeURIComponent(post?.title || '')

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
    }

    const shareUrl = shareUrls[platform]
    if (!shareUrl) return

    window.open(shareUrl, '_blank', 'width=600,height=600')
  }

  if (!post) {
    return (
      <div className="flex items-center justify-center px-6 py-32">
        <div className="text-center space-y-4">
          <h1 className="text-xl font-semibold">Post not found</h1>
          <Link to="/blogs" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors">
            <ArrowLeft className="w-4 h-4" />
            Back to blogs
          </Link>
        </div>
      </div>
    )
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
                <button
                  className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5 rounded-lg hover:bg-muted/50"
                >
                  <Share2 className="w-4 h-4" />
                  <span className="hidden sm:inline">Share</span>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent sideOffset={8} className="w-48">
                <DropdownMenuItem onClick={handleCopyLink} className="flex items-center gap-3">
                  {copied ? <Check className="w-4 h-4 text-green-500" /> : <LinkIcon className="w-4 h-4" />}
                  <span>{copied ? 'Copied!' : 'Copy link'}</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleShare('twitter')} className="flex items-center gap-3">
                  <Twitter className="w-4 h-4" />
                  <span>Share on Twitter</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleShare('linkedin')} className="flex items-center gap-3">
                  <Linkedin className="w-4 h-4" />
                  <span>Share on LinkedIn</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleShare('facebook')} className="flex items-center gap-3">
                  <Facebook className="w-4 h-4" />
                  <span>Share on Facebook</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleShare('whatsapp')} className="flex items-center gap-3">
                  <MessageCircle className="w-4 h-4" />
                  <span>Share on WhatsApp</span>
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleShare('reddit')} className="flex items-center gap-3">
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/>
                  </svg>
                  <span>Share on Reddit</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </div>
          </DropdownMenu>
        </div>

        {/* Article Header */}
        <header className="mb-10 animate-slide-up" style={{ animationDelay: '50ms' }}>
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
        <article className="mb-12 animate-slide-up" style={{ animationDelay: '100ms' }}>
          <MarkdownRenderer content={post.content} />
        </article>

        {/* Footer */}
        <footer className="pt-8 border-t border-border animate-slide-up" style={{ animationDelay: '150ms' }}>
          <div className="flex items-center justify-between">
            <Link
              to="/blogs"
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors group"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform duration-200" />
              All articles
            </Link>
            <button
              onClick={handleCopyLink}
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {copied ? <Check className="w-4 h-4 text-green-500" /> : <Share2 className="w-4 h-4" />}
              {copied ? 'Copied!' : 'Share'}
            </button>
          </div>
        </footer>
      </div>
    </div>
  )
}