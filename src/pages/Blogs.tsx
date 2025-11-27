import { Link } from "react-router-dom";
import posts from "../data/posts";
import { ArrowLeft, ArrowRight } from "../components/Icons";

type Post = {
  id: string;
  title: string;
  excerpt?: string;
  date?: string;
};

function PostItem({ post, index }: { post: Post; index: number }) {
  const delay = `${(index + 1) * 50}ms`;

  return (
    <Link
      key={post.id}
      to={`/blogs/${post.id}`}
      className="group flex items-start justify-between gap-6 py-5 border-b border-border hover:bg-muted/30 -mx-3 px-3 rounded-lg transition-all duration-200 animate-slide-up"
      style={{ animationDelay: delay }}
    >
      <div className="flex-1 min-w-0">
        <h2 className="font-medium text-foreground mb-1.5 group-hover:text-muted-foreground transition-colors duration-200">
          {post.title}
        </h2>
        {post.excerpt && (
          <p className="text-sm text-muted-foreground leading-relaxed">{post.excerpt}</p>
        )}
        <div className="flex items-center gap-3 shrink-0 pt-3 text-sm text-muted-foreground font-mono">
          {post.date}
          <ArrowRight aria-hidden className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all duration-200" />
        </div>
      </div>
    </Link>
  );
}

export default function Blogs() {
  return (
    <div className="animate-fade-in">
      <div className="max-w-[680px] mx-auto px-6 py-16 md:py-24">
        {/* Back link */}
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors mb-12 group"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform duration-200" aria-hidden />
          <span>Home</span>
        </Link>

        {/* Header */}
        <header className="mb-12 animate-slide-up">
          <h1 className="text-2xl font-semibold tracking-tight mb-2">Blogs</h1>
          <p className="text-muted-foreground text-sm">
            Welcome to my blog!
            <br />
            I'm finally getting myself to start publishing some of the notes,
            ideas, and writeups that usually just sit buried somewhere in my
            drive. Here, I'll be writing about my research, work, and
            interests across deep learning, foundation models, and AI more
            broadly. The posts might be a bit chaotic and messy but I hope
            they're still useful or interesting. I really appreciate your
            patience (and feedback!) as I get better at sharing my thoughts
            more openly.
          </p>
        </header>

        {/* Posts */}
        <div className="space-y-0">
          {posts.map((post, idx) => (
            <PostItem post={post as Post} index={idx} key={post.id} />
          ))}
        </div>
      </div>
    </div>
  );
}
