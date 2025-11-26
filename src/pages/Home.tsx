import { Link } from "react-router-dom"
import { Mail, Github, Twitter, Linkedin, GraduationCap, ArrowRight } from 'lucide-react'
import news from "../data/news"

export default function Home() {
  return (
    // <CHANGE> Removed min-h-screen, let flex-1 handle height
    <div className="animate-fade-in">
      <div className="max-w-[680px] mx-auto px-6 py-16 md:py-24">
        
        {/* Hero - Name & Image */}
        <header className="mb-16 animate-slide-up">
          <div className="flex items-start gap-6 mb-6">
            {/* <CHANGE> Added subtle hover animation to profile image */}
            <div className="w-20 h-20 rounded-full overflow-hidden bg-muted shrink-0 ring-2 ring-border hover:ring-muted-foreground transition-all duration-300">
              <img
                src="/assets/images/profile.webp"
                alt="Gauri Gupta"
                className="w-full h-full object-cover"
              />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight mb-1">Gauri Gupta</h1>
              <p className="text-muted-foreground text-sm">San Francisco, CA</p>
            </div>
          </div>
          
          {/* Social Links */}
          <div className="flex items-center gap-4">
            {[
              { icon: Mail, href: "mailto:gaurigupta.iitd@gmail.com", label: "Email" },
              { icon: Github, href: "https://github.com/gaurigupta19", label: "GitHub" },
              { icon: Twitter, href: "https://x.com/gauri__gupta", label: "Twitter" },
              { icon: Linkedin, href: "https://www.linkedin.com/in/gauri-gupta-115567162", label: "LinkedIn" },
              { icon: GraduationCap, href: "https://scholar.google.com/citations?user=SPaOg4cAAAAJ&hl=en", label: "Google Scholar" },
            ].map((link) => {
              const Icon = link.icon
              return (
                <a
                  key={link.href}
                  href={link.href}
                  target={link.href.startsWith("mailto:") ? undefined : "_blank"}
                  rel="noreferrer"
                  aria-label={link.label}
                  className="text-muted-foreground hover:text-foreground hover:scale-110 transition-all duration-200"
                >
                  <Icon className="w-[18px] h-[18px]" />
                </a>
              )
            })}
          </div>
        </header>

        {/* About Section */}
        <section className="mb-16 animate-slide-up" style={{ animationDelay: '100ms' }}>
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-6">About</h2>
          <div className="space-y-5 text-foreground/90 leading-[1.8]">
            <p>
              Hi! I am currently building my company researching post-training
              data for advancing model capabilities. We are currently in stealth
              mode and will share more soon.
            </p>
            <p>
              Before that, most recently I was at{" "}
              <a href="https://parallel.ai/" className="content-link">Parallel</a> (in its very early 10
              employees) with{" "}
              <a href="https://www.linkedin.com/in/paragagr/" className="content-link">
                Parag Agrawal,
              </a>{" "}
              where I built and released our SOTA web{" "}
              <a href="https://parallel.ai/blog/deep-research" className="content-link">
                deep-research agents
              </a>{" "}
              and spent a lot of time understanding agentic architecture and
              building datasets and evals. Prior to that, I was doing my PhD at
              MIT Media Lab with{" "}
              <a href="https://www.media.mit.edu/people/raskar/overview/" className="content-link">
                Prof. Ramesh Raskar
              </a>{" "}
              on Deep Learning and Foundational Model research where I published
              a bunch of research in understanding models and data distributions
              for model training with my amazing collaborators and labmates.
              During my PhD, I had the pleasure to intern at Apple MLR, doing
              cool research with amazing researchers{" "}
              <a
                href="https://scholar.google.com/citations?user=kjMNMLkAAAAJ&hl=en"
                target="_blank"
                rel="noreferrer"
                className="content-link"
              >
                Navdeep Jaitly
              </a>
              {", "}
              <a
                href="https://scholar.google.com/citations?user=Sv2TGqsAAAAJ&hl=en"
                target="_blank"
                rel="noreferrer"
                className="content-link"
              >
                Josh Susskind
              </a>
              {", and "}
              <a
                href="https://bengio.abracadoudou.com/"
                target="_blank"
                rel="noreferrer"
                className="content-link"
              >
                Samy Bengio
              </a>{" "}
              where I mostly worked on text-image diffusion models. I decided to
              drop out of my PhD after a couple years, which was not an easy
              decision but I realized I wanted to do more industry related
              product driven research.
            </p>
            <p>
              Prior to MIT, I did my bachelors from India in Mathematics and
              Computing from IIT Delhi. During my undergrad years, I had been
              fortunate to pursue research with amazing people and industry labs
              including Adobe MDSR team, Amazon IML team and{" "}
              <a
                href="https://www.kcl.ac.uk/people/peter-jossen"
                target="_blank"
                rel="noreferrer"
                className="content-link"
              >
                Peter Jossen
              </a>{" "}
              at ETH Zurich.
            </p>
            <p>
              Fun fact: I love travelling and spent ~4 months during my
              undergrad exchange semester backpacking across Europe, sleeping in
              trains and living off of bread and nutella :P
            </p>
          </div>
        </section>

        {/* News Section */}
        <section className="mb-16 animate-slide-up" style={{ animationDelay: '200ms' }}>
          <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-6">News</h2>
          <div className="space-y-4">
            {news.map((n, idx) => (
              <div 
                key={idx} 
                className="flex gap-6 group hover:bg-muted/30 -mx-3 px-3 py-2 rounded-lg transition-colors duration-200"
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
        </section>

        {/* Navigation Links */}
        {/* <CHANGE> Changed "Writing" to "Blogs" */}
        <section className="animate-slide-up" style={{ animationDelay: '300ms' }}>
          <div className="flex flex-col">
            <Link 
              to="/blogs" 
              className="group flex items-center justify-between py-4 border-t border-border hover:bg-muted/30 -mx-3 px-3 rounded-lg transition-all duration-200"
            >
              <span className="font-medium">Blogs</span>
              <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all duration-200" />
            </Link>
            <Link 
              to="/news" 
              className="group flex items-center justify-between py-4 border-t border-b border-border hover:bg-muted/30 -mx-3 px-3 rounded-lg transition-all duration-200"
            >
              <span className="font-medium">All Updates</span>
              <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground group-hover:translate-x-1 transition-all duration-200" />
            </Link>
          </div>
        </section>
      </div>
    </div>
  )
}
