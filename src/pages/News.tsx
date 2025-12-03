import { Link } from "react-router-dom"
import news from "../data/news"
import { ArrowLeft } from "../components/Icons"

export default function NewsPage() {
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
          <h1 className="text-2xl font-semibold tracking-tight mb-2">News</h1>
          <p className="text-muted-foreground text-sm">
            Updates and announcements.
          </p>
        </header>

        {/* News List */}
        <div className="space-y-0">
          {news.map((n, idx) => (
            <div 
              key={idx} 
              className="flex gap-6 py-4 border-b border-border last:border-0 hover:bg-muted/30 -mx-3 px-3 rounded-lg transition-all duration-200 animate-slide-up"
              style={{ animationDelay: `${(idx + 1) * 50}ms` }}
            >
              <div className="w-24 shrink-0 text-sm text-muted-foreground font-mono">
                {n.date}
              </div>
              <div 
                className="flex-1 text-foreground/90 leading-[1.7] news-content"
                dangerouslySetInnerHTML={{ __html: n.content }} 
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
