import { Link } from "react-router-dom";
import ThemeToggle from "./ThemeToggle";

export default function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 w-full backdrop-blur-sm bg-background/95 border-b border-border/40">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 py-4 flex items-center justify-between">
        {/* Logo/Brand */}
        <Link
          to="/"
          className="text-lg font-bold tracking-tight transition-opacity group"
        >
          <span className="inline-block">Gauri Gupta</span>
          <span className="block text-xs font-normal text-muted-foreground mt-0.5">
            Founder @
            <a
              href="https://www.neosigma.ai/"
              target="_blank"
              rel="noreferrer"
              className="hover:underline"
            >
              NeoSigma
            </a>{" "}
          </span>
        </Link>

        {/* Navigation Links */}
        <div className="flex items-center gap-8">
          <div className="hidden sm:flex gap-8">
            <Link
              to="/"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors relative group"
            >
              Home
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300" />
            </Link>
            <Link
              to="/blogs"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors relative group"
            >
              Blogs
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300" />
            </Link>
            <Link
              to="/news"
              className="text-sm font-medium text-foreground/70 hover:text-foreground transition-colors relative group"
            >
              News
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300" />
            </Link>
          </div>

          {/* Theme Toggle */}
          <ThemeToggle />
        </div>
      </nav>
    </header>
  );
}
