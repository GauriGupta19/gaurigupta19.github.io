import { Link } from "react-router-dom"
import posts from "../data/posts"
import { ArrowLeft, ArrowRight } from 'lucide-react'

// <CHANGE> Changed "Writing" to "Blogs", added animations
export default function Blogs() {
  return (
    <div className="animate-fade-in">
      <div className="max-w-[680px] mx-auto px-6 py-16 md:py-24">
        {/* Back link */}
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-12 group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform duration-200" />
          Home
        </Link>

        {/* Header */}
        <header className="mb-12 animate-slide-up">
          <h1 className="text-2xl font-semibold tracking-tight mb-2">Blogs</h1>
          <p className="text-muted-foreground text-sm">
            Notes on AI, machine learning, and engineering.
          </p>
        </header>

        {/* Posts */}
        <div className="space-y-0">
          {posts.map((post, idx) => (
            <Link
              key={post.id}
              to={`/blogs/${post.id}`}
              className="group flex items-start justify-between gap-6 py-5 border-b border-border hover:bg-muted/30 -mx-3 px-3 rounded-lg transition-all duration-200 animate-slide-up"
              style={{ animationDelay: `${(idx + 1) * 50}ms` }}
            >
              <div className="flex-1 min-w-0">
                <h2 className="font-medium text-foreground mb-1.5 group-hover:text-muted-foreground transition-colors duration-200">
                  {post.title}
                </h2>
                <p className="text-sm text-muted-foreground line-clamp-2 leading-relaxed">
                  {post.excerpt}
                </p>
              </div>
              <div className="flex items-center gap-3 shrink-0 pt-1">
                <span className="text-xs text-muted-foreground hidden sm:block font-mono">{post.date}</span>
                <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all duration-200" />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}
