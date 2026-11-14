"""Browser-side MP4/container and first-frame preflight checks."""

UPLOAD_ERROR_PREFLIGHT_JS = r"""    // Inspect MP4 top-level boxes before allowing Gradio to handle a local file.
    // This catches missing/truncated moov atoms without uploading the damaged
    // payload or triggering Gradio's native "Video not playable" notification.
    // Other containers and all server-side inputs are checked by ffprobe.
    const inspectMp4 = async (file) => {
        if (!/\.(mp4|m4v)$/i.test(file.name)) return null;
        let offset = 0;
        let hasFtyp = false;
        let hasMoov = false;
        let boxes = 0;
        while (offset + 8 <= file.size && boxes < 4096) {
            const data = new DataView(await file.slice(offset, offset + 16).arrayBuffer());
            const type = String.fromCharCode(...new Uint8Array(data.buffer, 4, 4));
            let length = data.getUint32(0);
            let headerLength = 8;
            if (length === 1) {
                if (data.byteLength < 16) return hasFtyp ? "truncated_file" : "invalid_container";
                const extended = data.getBigUint64(8);
                if (extended > BigInt(Number.MAX_SAFE_INTEGER)) return "invalid_container";
                length = Number(extended);
                headerLength = 16;
            } else if (length === 0) {
                length = file.size - offset;
            }
            if (length < headerLength) return "invalid_container";
            if (offset + length > file.size) return hasFtyp ? "truncated_file" : "invalid_container";
            if (type === "ftyp") hasFtyp = true;
            if (type === "moov") hasMoov = true;
            offset += length;
            boxes += 1;
        }
        // After a valid ftyp, trailing 1-7 bytes are a cut-off atom header.
        if (hasFtyp && offset < file.size && file.size - offset < 8) return "truncated_file";
        return hasFtyp && hasMoov && offset === file.size ? null : "invalid_container";
    };

    // For a locally selected MP4, ask the browser to decode one frame before
    // passing the file to Gradio. This avoids its generic red "not playable"
    // toast when the container is valid but the video payload is unreadable.
    // A browser that cannot decode a particular codec is not proof of damage:
    // only MEDIA_ERR_DECODE (3) rejects here. FFmpeg is the server authority.
    const inspectFirstFrame = (file) => {
        if (!/\.(mp4|m4v)$/i.test(file.name)) return Promise.resolve(null);
        return new Promise((resolve) => {
            const video = document.createElement("video");
            video.preload = "auto";
            video.muted = true;
            video.playsInline = true;
            const url = URL.createObjectURL(file);
            let finished = false;
            let timeout = null;
            const finish = (kind) => {
                if (finished) return;
                finished = true;
                window.clearTimeout(timeout);
                video.pause();
                video.removeAttribute("src");
                video.load();
                URL.revokeObjectURL(url);
                resolve(kind);
            };
            video.addEventListener("loadeddata", () => finish(null), {once: true});
            video.addEventListener("error", () => {
                finish(video.error?.code === 3 ? "undecodable_video" : null);
            }, {once: true});
            // Some browsers cannot decode a supported server-side codec; in
            // that case, let the server decide rather than reject the file.
            timeout = window.setTimeout(() => finish(null), 4000);
            video.src = url;
            video.load();
        });
    };

    const inspectFile = async (file) => {
        if (!file) return null;
        if (file.size === 0) return "empty_file";
        try {
            // A read failure is different from a malformed media container.
            await file.slice(0, 1).arrayBuffer();
            const containerError = await inspectMp4(file);
            if (containerError) return containerError;
            return await inspectFirstFrame(file);
        } catch (_) {
            return "upload";
        }
    };

"""

__all__ = ["UPLOAD_ERROR_PREFLIGHT_JS"]
