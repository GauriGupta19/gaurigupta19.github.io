import { Link } from "react-router-dom";
import { useState } from "react";
import ThemeToggle from "./ThemeToggle";
import { Menu, X } from "./Icons";

export default function Header() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const handleLinkClick = () => {
    setMobileMenuOpen(false);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 w-full backdrop-blur-sm bg-background/95 border-b border-border/40">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 py-3 sm:py-4 flex items-center justify-between">
        {/* Logo/Brand */}
        <Link
          to="/"
          onClick={handleLinkClick}
          className="text-base sm:text-lg font-bold tracking-tight transition-opacity group"
        >
          <span className="inline-block">Gauri Gupta</span>
          <span className="hidden sm:block text-xs font-normal text-muted-foreground mt-0.5">
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

        {/* Desktop Navigation Links */}
        <div className="hidden sm:flex items-center gap-6 lg:gap-8">
          <div className="flex gap-6 lg:gap-8">
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

        {/* Mobile: Hamburger + Theme Toggle */}
        <div className="flex sm:hidden items-center gap-2">
          <ThemeToggle />
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 text-foreground hover:bg-muted/50 rounded-lg transition-colors touch-manipulation"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <X className="w-5 h-5" />
            ) : (
              <Menu className="w-5 h-5" />
            )}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="sm:hidden border-t border-border/40 bg-background/95 backdrop-blur-sm">
          <div className="px-4 py-4 space-y-1">
            <Link
              to="/"
              onClick={handleLinkClick}
              className="block px-4 py-3 text-sm font-medium text-foreground/70 hover:text-foreground hover:bg-muted/50 rounded-lg transition-colors touch-manipulation"
            >
              Home
            </Link>
            <Link
              to="/blogs"
              onClick={handleLinkClick}
              className="block px-4 py-3 text-sm font-medium text-foreground/70 hover:text-foreground hover:bg-muted/50 rounded-lg transition-colors touch-manipulation"
            >
              Blogs
            </Link>
            <Link
              to="/news"
              onClick={handleLinkClick}
              className="block px-4 py-3 text-sm font-medium text-foreground/70 hover:text-foreground hover:bg-muted/50 rounded-lg transition-colors touch-manipulation"
            >
              News
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
