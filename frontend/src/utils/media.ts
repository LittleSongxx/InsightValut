const PUBLIC_MINIO_PORT = '19100';
const IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg'];

function getPublicMinioOrigin() {
  if (typeof window === 'undefined') {
    return `http://127.0.0.1:${PUBLIC_MINIO_PORT}`;
  }

  const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
  const hostname = window.location.hostname || '127.0.0.1';
  return `${protocol}//${hostname}:${PUBLIC_MINIO_PORT}`;
}

export function normalizeAssetUrl(url?: string | null): string {
  const raw = (url ?? '').trim();
  if (!raw) return '';
  if (raw.startsWith('data:') || raw.startsWith('blob:')) return raw;

  let normalized = raw
    .replace('/knowledge-base-filesupload-images/', '/knowledge-base-files/upload-images/')
    .replace('/knowledge-base-filespdf_files/', '/knowledge-base-files/pdf_files/');

  normalized = normalized.replace(/^https?:\/\/minio:9000/i, getPublicMinioOrigin());
  normalized = normalized.replace(
    /(https?:\/\/[^/]+\/knowledge-base-files)(?=(?:upload-images|pdf_files)\/)/i,
    '$1/',
  );

  if (normalized.startsWith('//')) {
    const protocol = typeof window !== 'undefined' ? window.location.protocol : 'http:';
    return `${protocol}${normalized}`;
  }

  return normalized;
}

function isRenderableImageUrl(url: string): boolean {
  if (!url) return false;
  if (url.startsWith('data:image/') || url.startsWith('blob:')) return true;
  if (/\s/.test(url)) return false;

  try {
    const parsed = new URL(url, typeof window !== 'undefined' ? window.location.origin : 'http://127.0.0.1');
    const path = parsed.pathname.toLowerCase();
    return IMAGE_EXTENSIONS.some((ext) => path.endsWith(ext));
  } catch {
    return false;
  }
}

export function normalizeAssetUrls(urls?: Array<string | null> | null): string[] {
  const unique = new Set<string>();

  for (const url of urls || []) {
    const normalized = normalizeAssetUrl(url);
    if (normalized && isRenderableImageUrl(normalized)) {
      unique.add(normalized);
    }
  }

  return Array.from(unique);
}
