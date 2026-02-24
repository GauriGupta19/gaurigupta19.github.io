import { Link } from "react-router-dom";
import {
  MapPin,
  Mail,
  Twitter,
  Linkedin,
  GraduationCap,
  News,
  Blogs,
} from "./Icons";

interface SidebarProps {
  isMobileMenuOpen?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ isMobileMenuOpen = false, onClose }: SidebarProps) {
  const socialLinks = [
    { icon: MapPin, label: "San Francisco, CA", href: null },
    { icon: Mail, label: "Email", href: "mailto:gaurigupta.iitd@gmail.com" },
    { icon: Twitter, label: "X", href: "https://x.com/gauri__gupta" },
    {
      icon: Linkedin,
      label: "LinkedIn",
      href: "https://www.linkedin.com/in/gauri-gupta-115567162",
    },
    {
      icon: GraduationCap,
      label: "Google Scholar",
      href: "https://scholar.google.com/citations?user=SPaOg4cAAAAJ&hl=en",
    },
  ];

  // Mobile full-screen menu overlay (when hamburger is clicked)
  if (isMobileMenuOpen) {
    return (
      <div className="flex flex-col items-center py-8 px-6 overflow-y-auto h-full">
        {/* Profile Image */}
        <div className="w-64 h-64 rounded-lg overflow-hidden bg-muted mb-6">
          <img
            src="/assets/images/profile.webp"
            alt="Gauri Gupta"
            className="w-full h-full object-cover"
            loading="lazy"
          />
        </div>

        {/* Name & Bio */}
        <div className="text-center mb-8">
          <h3 className="text-2xl font-bold mb-2">Gauri Gupta</h3>
          <p className="text-base text-muted-foreground">
            Founder @{" "}
            <a
              href="https://www.neosigma.ai/"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:underline"
            >
              NeoSigma
            </a>
          </p>
        </div>

        {/* Social Links & Navigation */}
        <div className="w-full max-w-sm space-y-1">
          {socialLinks.map((link) => {
            const Icon = link.icon;
            if (!link.href) {
              return (
                <div
                  key={link.label}
                  className="flex items-center gap-3 text-sm py-2 px-4"
                >
                  <Icon className="w-4 h-4 text-muted-foreground" />
                  <span className="text-muted-foreground">{link.label}</span>
                </div>
              );
            }
            return (
              <a
                key={link.label}
                href={link.href}
                target={link.href.startsWith("mailto:") ? undefined : "_blank"}
                rel="noreferrer"
                className="flex items-center gap-3 text-sm py-2 px-4 hover:bg-muted/50 rounded-lg transition-all group"
                onClick={onClose}
              >
                <Icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
                <span>{link.label}</span>
              </a>
            );
          })}
          
          {/* Navigation Links */}
          <Link
            to="/blogs"
            onClick={onClose}
            className="flex items-center gap-3 text-sm py-2 px-4 hover:bg-muted/50 rounded-lg transition-all group"
          >
            <Blogs className="w-4 h-4 group-hover:scale-110 transition-transform" />
            <span>Blogs</span>
          </Link>
          <Link
            to="/news"
            onClick={onClose}
            className="flex items-center gap-3 text-sm py-2 px-4 hover:bg-muted/50 rounded-lg transition-all group"
          >
            <News className="w-4 h-4 group-hover:scale-110 transition-transform" />
            <span>News</span>
          </Link>
        </div>
      </div>
    );
  }

  // Regular sidebar (both mobile and desktop)
  return (
    <aside className="flex flex-col gap-6 md:gap-8 w-full md:w-72 shrink-0 items-center md:items-start md:sticky md:top-20 md:h-[calc(100vh-5rem)] md:overflow-y-auto">
      {/* Profile Image */}
      <div className="h-80 w-80 md:h-72 md:w-72 rounded-lg overflow-hidden bg-muted">
        <img
          src="/assets/images/profile.webp"
          alt="Gauri Gupta"
          className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
          loading="lazy"
        />
      </div>

      {/* About Section */}
      <div className="text-center md:text-left">
        <h3 className="text-2xl font-bold mb-2">Gauri Gupta</h3>
        <p className="text-sm text-muted-foreground">
          Founder @{" "}
          <a
            href="https://www.neosigma.ai/"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:underline"
          >
            NeoSigma
          </a>
        </p>
      </div>

      {/* Social Links */}
      <div className="space-y-4 w-full max-w-xs md:max-w-none flex flex-col items-center md:items-start">
        {socialLinks.map((link) => {
          const Icon = link.icon;
          if (!link.href) {
            return (
              <div
                key={link.label}
                className="flex items-center gap-3 text-sm"
              >
                <Icon className="w-4 h-4 text-muted-foreground" />
                <span className="text-muted-foreground">{link.label}</span>
              </div>
            );
          }
          return (
            <a
              key={link.label}
              href={link.href}
              target={link.href.startsWith("mailto:") ? undefined : "_blank"}
              rel="noreferrer"
              className="flex items-center gap-3 text-sm hover:text-primary transition-colors group"
            >
              <Icon className="w-4 h-4 group-hover:scale-110 transition-transform" />
              <span>{link.label}</span>
            </a>
          );
        })}
      </div>

      {/* Navigation Links */}
      <div className="flex flex-col gap-3 w-full max-w-xs md:max-w-none items-center md:items-start">
        <Link
          to="/blogs"
          className="flex items-center gap-3 text-sm hover:text-primary transition-colors group"
        >
          <Blogs className="w-4 h-4 group-hover:scale-110 transition-transform" />
          <span>Blogs</span>
        </Link>
        <Link
          to="/news"
          className="flex items-center gap-3 text-sm hover:text-primary transition-colors group"
        >
          <News className="w-4 h-4 group-hover:scale-110 transition-transform" />
          <span>News</span>
        </Link>
      </div>
    </aside>
  );
}