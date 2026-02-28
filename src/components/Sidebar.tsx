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
  // Mobile full-screen menu overlay (when hamburger is clicked)
  if (isMobileMenuOpen) {
    return (
      <div className="flex flex-col items-center py-8 px-6 overflow-y-auto h-full">
        <div className="w-64 h-64 rounded-xl overflow-hidden bg-muted mb-6 ring-1 ring-border/50">
          <img
            src="/assets/images/profile.webp"
            alt="Gauri Gupta"
            className="w-full h-full object-cover"
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
    <aside className="flex flex-col gap-5 md:gap-6 w-full md:w-72 shrink-0 items-center md:items-start md:sticky md:top-20 md:h-[calc(100vh-5rem)] md:overflow-hidden">
      {/* Profile Image */}
      <div className="h-72 w-72 md:h-64 md:w-64 rounded-xl overflow-hidden bg-muted ring-1 ring-border/50">
        <img
          src="/assets/images/profile.webp"
          alt="Gauri Gupta"
          className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
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
      <div className="flex gap-2">
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
    </aside>
  );
}