import { Toaster as Sonner, type ToasterProps } from "sonner"

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="dark"
      position="top-right"
      className="toaster group"
      style={
        {
          "--normal-bg": "#1e293b",
          "--normal-text": "#ffffff",
          "--normal-border": "rgba(255, 255, 255, 0.08)",
          "--border-radius": "0.5rem",
        } as React.CSSProperties
      }
      {...props}
    />
  )
}

export { Toaster }
