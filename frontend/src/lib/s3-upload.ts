/**
 * Upload a local file directly to AWS S3 using a short-lived presigned PUT URL.
 * Emits upload progress percentages and never exposes AWS credentials to the browser.
 */
export function uploadFileToS3(
  file: File,
  uploadUrl: string,
  contentType: string,
  onProgress?: (percent: number) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', uploadUrl, true);
    xhr.setRequestHeader('Content-Type', contentType);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        const percent = Math.round((e.loaded / e.total) * 100);
        onProgress(percent);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        if (onProgress) onProgress(100);
        resolve();
      } else {
        let msg = xhr.statusText || 'Unknown Error';
        try {
          if (xhr.responseText) {
            const match = xhr.responseText.match(/<Message>(.*?)<\/Message>/);
            if (match && match[1]) {
              msg = match[1];
            }
          }
        } catch {
          // ignore
        }
        reject(
          new Error(
            `S3 upload failed with HTTP ${xhr.status}: ${msg}`
          )
        );
      }
    };

    xhr.onerror = () => {
      reject(new Error('Network error occurred during direct S3 upload. Check CORS settings.'));
    };

    xhr.ontimeout = () => {
      reject(new Error('S3 upload timed out.'));
    };

    xhr.send(file);
  });
}
