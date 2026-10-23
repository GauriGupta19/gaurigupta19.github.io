import { Link } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import news from "../data/news";

export default function Home() {
  return (
    // Two-column layout: fixed left sidebar (1/3) and scrollable right content (2/3)
    <div className="animate-fade-in home-page">
      <div className="md:flex">
        {/* Mobile: Sidebar at top */}
        <aside className="md:hidden w-full border-b border-border bg-background pt-4">
          <div className="px-4 pb-8 max-w-2xl mx-auto">
            <Sidebar />
          </div>
        </aside>

        {/* Desktop: fixed sidebar on medium+ screens */}
        <aside className="home-sidebar hidden md:block md:fixed md:top-16 md:bottom-0 md:w-1/3 md:overflow-y-auto md:pt-8">
          <div className="h-full p-6 flex justify-center items-start">
            <Sidebar />
          </div>
        </aside>

        {/* Right: main content - add left margin equal to sidebar width on md+ */}
        <main className="w-full md:ml-[33.3333%] md:w-2/3">
          <div className="max-w-[680px] mx-auto px-4 sm:px-6 py-8 sm:py-12 md:py-24">
            {/* About Section */}
            <section
              id="about"
              className="mb-12 sm:mb-16 animate-slide-up"
              style={{ animationDelay: "100ms" }}
            >
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-4 sm:mb-6">
                About
              </h2>
              <p className="text-[15px] sm:text-base text-foreground/90 leading-relaxed mb-6">
                A little about me and the work that brought me here.
              </p>
              <div className="career-timeline-scroll" role="region" aria-label="Career timeline, newest first" tabIndex={0}>
                <ol className="career-timeline">
                  {/* Add the next chapter here as the story evolves, newest first. */}
                  <li className="career-milestone career-milestone-current">
                    <span className="career-stage">Now</span>
                    <h3>NeoSigma</h3>
                    <p>I eventually took a leap of faith and started my own company with my amazing co-founder, Ritvik Kapila. We are building <a href="https://neosigma.ai" target="_blank" rel="noreferrer" className="content-link">NeoSigma</a>, an infrastructure for self-improving agentic systems that learn directly from experience. I’m excited for this journey and can’t wait to share more.</p>
                  </li>
                  <li className="career-milestone">
                    <h3>Parallel</h3>
                    <p>I was a founding engineer at <a href="https://parallel.ai/" className="content-link">Parallel</a> (building the intelligence layer for the modern web) founded by <a href="https://www.linkedin.com/in/paragagr/" className="content-link">Parag Agrawal</a>, the former CEO of Twitter. As one of the first ten employees, I helped build the core agentic technology and team from the ground up, including <a href="https://parallel.ai/blog/deep-research" className="content-link">Parallel’s deep-research agents</a>. I spent much of my time on agent quality research: understanding how agents behave at scale and building evals that capture that.</p>
                    <p>I loved the early 0-to-1 journey: bringing a vision to life as a product, talking to customers, and bringing together an incredible team. It gave me a close-up view of what it takes to build a company and the chance to help shape it along the way.</p>
                  </li>
                  <li className="career-milestone">
                    <h3>MIT Media Lab</h3>
                    <p>I pursued PhD research at MIT Media Lab with <a href="https://www.media.mit.edu/people/raskar/overview/" className="content-link">Prof. Ramesh Raskar</a>, working on deep learning and foundation models. I published research on understanding models and data distributions for model training with my amazing collaborators and labmates. You can find more of my research on <a href="https://www.media.mit.edu/people/gaurii/overview/" className="content-link">my MIT profile</a>.</p>
                    <p>During my PhD, I had the pleasure of interning at Apple MLR, doing research with <a href="https://scholar.google.com/citations?user=kjMNMLkAAAAJ&hl=en" className="content-link">Navdeep Jaitly</a>, <a href="https://scholar.google.com/citations?user=Sv2TGqsAAAAJ&hl=en" className="content-link">Josh Susskind</a>, and <a href="https://bengio.abracadoudou.com/" className="content-link">Samy Bengio</a>, where I worked on studying and improving correlation in text-image diffusion models.</p>
                    <p>I eventually decided to drop out of my PhD after a couple of years. It wasn’t an easy decision, but I realized I wanted to be closer to industry-driven research: solving challenging problems at scale and seeing that work have an immediate impact on customers. That led me to join <a href="https://parallel.ai/" className="content-link">Parallel</a>.</p>
                  </li>
                  <li className="career-milestone">
                    <h3>IIT Delhi</h3>
                    <p>I did my bachelor’s in Math &amp; CS at IIT Delhi. I’d been a math nerd pretty much my whole life :P</p>
                    <p>I explored abstract mathematics through problems in number theory during my internship at ETH Zurich with <a href="https://www.kcl.ac.uk/people/peter-jossen" target="_blank" rel="noreferrer" className="content-link">Prof. Peter Jossen</a>. Eventually, my curiosity led me into the world of machine learning, where I pursued research with amazing people at Adobe’s MDSR team and Amazon’s IML team, working on a range of deep learning problems.</p>
                    <p>Fun fact: I love travelling and spent about four months during my undergrad exchange semester backpacking across Europe, sleeping on trains and living off bread and Nutella :P</p>
                  </li>
                </ol>
              </div>
              <p className="text-xs text-muted-foreground mt-3">Scroll through my journey. Latest chapter first.</p>
            </section>

            <section id="investing" className="mb-12 sm:mb-16 animate-slide-up" style={{ animationDelay: "150ms" }} aria-labelledby="investing-heading">
              <h2 id="investing-heading" className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-4 sm:mb-6">
                Investing
              </h2>
              <p className="text-foreground/90 leading-[1.75] sm:leading-[1.8] text-[15px] sm:text-base mb-4">
                Backing ideas and founders that advance the frontier of technology.
              </p>
              <h3 className="text-sm text-muted-foreground mb-3">Select investments</h3>
              <ul className="list-disc pl-5 space-y-3 text-[15px] sm:text-base text-foreground/90 leading-relaxed marker:text-muted-foreground">
                <li><a href="https://warp.xyz/" target="_blank" rel="noreferrer" className="content-link">Warp Sciences</a>: AI and robotics to automate chemistry and accelerate drug discovery.</li>
                <li><a href="https://www.empiricalmachines.ai/" target="_blank" rel="noreferrer" className="content-link">Empirical Machines</a>: AI that learns from real-world experiments to accelerate scientific discovery.</li>
                <li><a href="https://www.coreauto.com/" target="_blank" rel="noreferrer" className="content-link">Core Automation</a>: An AI lab developing new learning algorithms and systems to automate research and knowledge work.</li>
                <li><a href="https://www.anthropic.com/" target="_blank" rel="noreferrer" className="content-link">Anthropic</a>: AI safety research and the Claude family of models and products.</li>
                <li><a href="https://optexity.com/" target="_blank" rel="noreferrer" className="content-link">Optexity</a>: Browser automation that turns web workflows into APIs, including for sites without existing integrations.</li>
              </ul>
            </section>

            {/* News Section (top 5) */}
            <section
              id="news"
              className="mb-8 sm:mb-12 animate-slide-up"
              style={{ animationDelay: "200ms" }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
                  News
                </h2>
                <Link
                  to="/news"
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors touch-manipulation min-h-11 px-3 -mr-3"
                >
                  Show more
                </Link>
              </div>

              <div className="space-y-3 sm:space-y-4">
                {news.slice(0, 5).map((n, idx) => (
                  <div
                    key={idx}
                    className="flex flex-col sm:flex-row gap-2 sm:gap-6 group hover:bg-muted/30 -mx-2 sm:-mx-3 px-2 sm:px-3 py-2 rounded-lg transition-colors duration-200"
                  >
                    <div className="w-full sm:w-24 shrink-0 text-xs sm:text-sm text-muted-foreground font-mono">
                      {n.date}
                    </div>
                    <div
                      className="flex-1 text-[15px] sm:text-base text-foreground/90 leading-[1.7] news-content"
                      dangerouslySetInnerHTML={{ __html: n.content }}
                    />
                  </div>
                ))}
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}
