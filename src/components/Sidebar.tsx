import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Mail, Twitter, Linkedin } from "./Icons";

interface SidebarProps {
  isMobileMenuOpen?: boolean;
  onClose?: () => void;
}

function TldrBio() {
  return (
    <>
      Building something new @{" "}
      <a
        href="https://x.com/neosigmaai"
        target="_blank"
        rel="noreferrer"
        className="content-link"
      >
        NeoSigma
      </a>
      {" • "}ex-{" "}
      <a
        href="https://x.com/p0"
        target="_blank"
        rel="noreferrer"
        className="content-link"
      >
        Parallel Web Systems
      </a>{" "}
      (early team) • MIT PhD dropout
    </>
  );
}

const socialButtons = [
  { icon: Twitter, href: "https://x.com/gauri__gupta", label: "X" },
  {
    icon: Linkedin,
    href: "https://www.linkedin.com/in/gauri-gupta-115567162",
    label: "LinkedIn",
  },
  { icon: Mail, href: "mailto:gaurigupta.iitd@gmail.com", label: "Email" },
];

export default function Sidebar({ isMobileMenuOpen = false, onClose }: SidebarProps) {
  const [activeSection, setActiveSection] = useState("about");

  useEffect(() => {
    let frame = 0;
    const updateSection = () => {
      const sections = ["about", "investing", "news"]
        .map(id => document.getElementById(id))
        .filter((section): section is HTMLElement => section !== null);
      const threshold = Math.min(window.innerHeight * 0.35, 220);
      const current = sections.filter(section => section.getBoundingClientRect().top <= threshold).at(-1);
      setActiveSection(current?.id ?? "about");
    };
    const handleScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(updateSection);
    };
    updateSection();
    window.addEventListener("scroll", handleScroll, { passive: true });
    window.addEventListener("resize", handleScroll);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", handleScroll);
      window.removeEventListener("resize", handleScroll);
    };
  }, []);

  // Mobile full-screen menu overlay (when hamburger is clicked)
  if (isMobileMenuOpen) {
    return (
      <div className="flex flex-col items-center py-8 px-6 overflow-y-auto h-full">
        <div className="w-64 h-64 rounded-xl overflow-hidden bg-muted mb-6 ring-1 ring-border/50">
          <img
            src="/assets/images/profile.webp"
            alt="Gauri Gupta"
            className="profile-portrait w-full h-full object-cover"
            loading="lazy"
          />
        </div>
        <div className="text-center mb-6">
          <h3 className="text-2xl font-bold mb-3">Gauri Gupta</h3>
          <p className="text-sm text-muted-foreground leading-relaxed max-w-xs">
            <TldrBio />
          </p>
        </div>
        <div className="flex gap-3">
          {socialButtons.map(({ icon: Icon, href, label }) => (
            <a
              key={label}
              href={href}
              target={href.startsWith("mailto:") ? undefined : "_blank"}
              rel="noreferrer"
              title={label}
              onClick={onClose}
              className="w-10 h-10 rounded-full bg-muted hover:bg-foreground hover:text-background flex items-center justify-center transition-all duration-200 group"
            >
              <Icon className="w-4 h-4" />
            </a>
          ))}
        </div>
      </div>
    );
  }

  // Regular sidebar (both mobile and desktop)
  return (
    <aside className="flex flex-col gap-5 md:gap-6 w-full md:w-72 shrink-0 items-center md:items-start md:sticky md:top-20 md:pb-6">
      {/* Profile Image */}
      <div className="h-72 w-72 md:h-64 md:w-64 shrink-0 rounded-xl overflow-hidden bg-muted ring-1 ring-border/50">
        <img
          src="/assets/images/profile.webp"
          alt="Gauri Gupta"
          className="profile-portrait w-full h-full object-cover"
          loading="lazy"
        />
      </div>

      {/* Name & TLDR Bio */}
      <div className="text-center md:text-left space-y-2">
        <h3 className="text-xl font-bold tracking-tight">Gauri Gupta</h3>
        <p className="text-[13px] text-muted-foreground leading-relaxed">
          <TldrBio />
        </p>
      </div>


      {/* Icon-only social buttons */}
      <div className="sidebar-socials flex gap-2">
        {socialButtons.map(({ icon: Icon, href, label }) => (
          <a
            key={label}
            href={href}
            target={href.startsWith("mailto:") ? undefined : "_blank"}
            rel="noreferrer"
            title={label}
            className="w-9 h-9 rounded-full bg-muted/80 hover:bg-foreground hover:text-background flex items-center justify-center transition-all duration-200 shrink-0"
          >
            <Icon className="w-[18px] h-[18px]" />
          </a>
        ))}
      </div>
      <nav aria-label="Table of contents" className="sidebar-contents w-full border-t border-border pt-4">
        <p className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2 text-center md:text-left">Contents</p>
        <ul className="flex flex-wrap justify-center md:flex-col md:items-start gap-x-5 gap-y-1 text-sm">
          <li><a href="#about" aria-current={activeSection === "about" ? "location" : undefined} className="inline-block py-1.5 text-muted-foreground hover:text-foreground transition-colors">About</a></li>
          <li><a href="#investing" aria-current={activeSection === "investing" ? "location" : undefined} className="inline-block py-1.5 text-muted-foreground hover:text-foreground transition-colors">Investing</a></li>
          <li><a href="#news" aria-current={activeSection === "news" ? "location" : undefined} className="inline-block py-1.5 text-muted-foreground hover:text-foreground transition-colors">News</a></li>
          <li><Link to="/blogs" className="inline-block py-1.5 text-muted-foreground hover:text-foreground transition-colors">Writing <span aria-hidden="true">↗</span></Link></li>
        </ul>
      </nav>
    </aside>
  );
}