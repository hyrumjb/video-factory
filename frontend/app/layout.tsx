import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Video Factory",
  description: "Create beautiful videos with zero effort",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
