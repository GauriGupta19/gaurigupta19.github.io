import React from "react"

type Props = React.PropsWithChildren<{ className?: string }>

export function Card({ children, className = "" }: Props) {
  return (
    <div className={`bg-card text-card-foreground border border-border rounded-md p-6 shadow-sm ${className}`}>
      {children}
    </div>
  )
}

export default Card
