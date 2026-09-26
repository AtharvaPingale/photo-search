import { photoUrls } from "./api";

/** Share the photo itself (not a link: the server is private) via the OS share sheet.
 *  Sends the 2048 px JPEG rendition, which every messaging app accepts, including
 *  for HEIC and RAW originals. Falls back to a download. */
export async function sharePhoto(photoId: string, name: string): Promise<"shared" | "downloaded" | "cancelled"> {
  const res = await fetch(photoUrls(photoId).display, { credentials: "same-origin" });
  const blob = await res.blob();
  const base = name.replace(/\.[^.]+$/, "") || "photo";
  const file = new File([blob], `${base}.jpg`, { type: "image/jpeg" });
  const nav = navigator as Navigator & { canShare?: (d: { files: File[] }) => boolean };
  if (nav.share && nav.canShare?.({ files: [file] })) {
    try {
      await nav.share({ files: [file] });
      return "shared";
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return "cancelled";
    }
  }
  downloadPhoto(photoId);
  return "downloaded";
}

export function downloadPhoto(photoId: string) {
  const a = document.createElement("a");
  a.href = photoUrls(photoId).download;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
