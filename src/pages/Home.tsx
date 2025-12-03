import { useState } from "react";
import Sidebar from "../components/Sidebar";
import news from "../data/news";

export default function Home() {
  const [showAll, setShowAll] = useState(false);

  return (
    // Two-column layout: fixed left sidebar (1/3) and scrollable right content (2/3)
    <div className="animate-fade-in">
      <div className="md:flex">
        {/* Left: fixed sidebar on medium+ screens */}
        <aside className="hidden md:block md:fixed md:top-16 md:bottom-0 md:w-1/3 md:overflow-auto">
          <div className="h-full p-6 flex justify-center items-center">
            <Sidebar />
          </div>
        </aside>

        {/* Right: main content - add left margin equal to sidebar width on md+ */}
        <main className="w-full md:ml-[33.3333%] md:w-2/3">
          <div className="max-w-[680px] mx-auto px-6 py-16 md:py-24">
            {/* About Section */}
            <section
              className="mb-16 animate-slide-up"
              style={{ animationDelay: "100ms" }}
            >
              <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-6">
                About
              </h2>
              <div className="space-y-5 text-foreground/90 leading-[1.8]">
                <p>
                  Hi! I am currently building my company researching
                  post-training data for advancing model capabilities. We are
                  currently in stealth mode and will share more soon.
                </p>
                <p>
                  Before that, most recently I was at{" "}
                  <a href="https://parallel.ai/" className="content-link">
                    Parallel
                  </a>{" "}
                  (in its very early 10 employees) with{" "}
                  <a
                    href="https://www.linkedin.com/in/paragagr/"
                    className="content-link"
                  >
                    Parag Agrawal,
                  </a>{" "}
                  where I built and released our SOTA web{" "}
                  <a
                    href="https://parallel.ai/blog/deep-research"
                    className="content-link"
                  >
                    deep-research agents
                  </a>{" "}
                  and spent a lot of time understanding agentic architecture and
                  building datasets and evals. Prior to that, I was doing my PhD
                  at MIT Media Lab with{" "}
                  <a
                    href="https://www.media.mit.edu/people/raskar/overview/"
                    className="content-link"
                  >
                    Prof. Ramesh Raskar
                  </a>{" "}
                  on Deep Learning and Foundational Model research where I
                  published a bunch of research in understanding models and data
                  distributions for model training with my amazing collaborators
                  and labmates. During my PhD, I had the pleasure to intern at
                  Apple MLR, doing cool research with amazing researchers{" "}
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
                  where I mostly worked on text-image diffusion models. I
                  decided to drop out of my PhD after a couple years, which was
                  not an easy decision but I realized I wanted to do more
                  industry related product driven research.
                </p>
                <p>
                  Prior to MIT, I did my bachelors from India in Mathematics and
                  Computing from IIT Delhi. During my undergrad years, I had
                  been fortunate to pursue research with amazing people and
                  industry labs including Adobe MDSR team, Amazon IML team and{" "}
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
                  undergrad exchange semester backpacking across Europe,
                  sleeping in trains and living off of bread and nutella :P
                </p>
              </div>
            </section>

            {/* News Section (top 5, expandable) */}
            <section
              className="mb-12 animate-slide-up"
              style={{ animationDelay: "200ms" }}
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
                  News
                </h2>
                <button
                  onClick={() => setShowAll((s) => !s)}
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showAll ? "Show less" : "Show more"}
                </button>
              </div>

              <div className="space-y-4">
                {(showAll ? news : news.slice(0, 5)).map((n, idx) => (
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
          </div>
        </main>
      </div>
    </div>
  );
}
