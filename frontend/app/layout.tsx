import type { Metadata } from "next";
import "./globals.css";
import FirebaseProvider from "@/components/FirebaseProvider";

export const metadata: Metadata = {
  title: "Lightfall",
  description: "Create beautiful videos with zero effort",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <FirebaseProvider>
          {children}
        </FirebaseProvider>
      </body>
    </html>
  );
}
