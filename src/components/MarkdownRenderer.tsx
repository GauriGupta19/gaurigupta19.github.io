import React from 'react'
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownRendererProps {
  content: string;
}

export default function MarkdownRenderer({ content }: MarkdownRendererProps) {
  return (
    <div className="prose">
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ ...props }) => (
            <h1 className="text-2xl font-semibold mt-12 mb-4 tracking-tight text-foreground" {...props} />
          ),
          h2: ({ ...props }) => (
            <h2 className="text-xl font-semibold mt-10 mb-4 tracking-tight text-foreground" {...props} />
          ),
          h3: ({ ...props }) => (
            <h3 className="text-lg font-semibold mt-8 mb-3 text-foreground" {...props} />
          ),
          h4: ({ ...props }) => (
            <h4 className="text-base font-semibold mt-6 mb-2 text-foreground" {...props} />
          ),
          p: ({ ...props }) => (
            <p className="mb-5 leading-[1.8] text-foreground/85" {...props} />
          ),
          a: ({ ...props }) => (
            <a 
              className="text-foreground underline underline-offset-[3px] decoration-muted-foreground/40 hover:decoration-foreground transition-colors" 
              target="_blank" 
              rel="noreferrer" 
              {...props} 
            />
          ),
          strong: ({ ...props }) => (
            <strong className="font-semibold text-foreground" {...props} />
          ),
          code: ({ className, children, ...props }: { className?: string; children?: React.ReactNode } & React.HTMLAttributes<HTMLElement>) => {
            const isBlock = className?.includes('language-');
            if (isBlock) {
              return <code className={className} {...props}>{children}</code>;
            }
            return (
              <code className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono text-foreground" {...props}>
                {children}
              </code>
            );
          },
          pre: ({ ...props }) => (
            <pre className="bg-muted p-4 rounded-lg overflow-x-auto my-6 text-sm" {...props} />
          ),
          // <CHANGE> Let CSS handle list styling - no custom rendering needed
          ul: ({ ...props }) => (
            <ul className="my-5 ml-4" {...props} />
          ),
          ol: ({ ...props }) => (
            <ol className="my-5 ml-4" {...props} />
          ),
          li: ({ ...props }) => (
            <li className="text-foreground/85 leading-[1.8]" {...props} />
          ),
          blockquote: ({ ...props }) => (
            <blockquote className="border-l-2 border-border pl-4 my-6 text-foreground/70 italic" {...props} />
          ),
          hr: ({ ...props }) => (
            <hr className="my-10 border-t border-border" {...props} />
          ),
          table: ({ ...props }) => (
            <div className="overflow-x-auto my-6">
              <table className="w-full border-collapse text-sm" {...props} />
            </div>
          ),
          th: ({ ...props }) => (
            <th className="border border-border bg-muted px-3 py-2 text-left font-semibold text-foreground" {...props} />
          ),
          td: ({ ...props }) => (
            <td className="border border-border px-3 py-2 text-foreground/85" {...props} />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
