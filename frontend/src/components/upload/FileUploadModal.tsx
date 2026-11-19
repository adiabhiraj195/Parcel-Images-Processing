'use client';

import React, { useState, useRef } from 'react';
import { Upload, X, Check, AlertCircle, RefreshCw, FileText } from 'lucide-react';
import { createBatch, completeBatchUploads, PresignedUploadTarget } from '@/lib/api';
import { uploadFileToS3 } from '@/lib/s3-upload';

interface FileUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onBatchCreated: (batchId: string) => void;
}

interface UploadingFile {
  file: File;
  id?: string;
  status: 'pending' | 'uploading' | 'completed' | 'failed';
  progress: number;
  error?: string;
  presigned?: PresignedUploadTarget;
}

export const FileUploadModal: React.FC<FileUploadModalProps> = ({
  isOpen,
  onClose,
  onBatchCreated,
}) => {
  const [batchName, setBatchName] = useState('');
  const [files, setFiles] = useState<UploadingFile[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFilesAdded = (incomingFiles: FileList | null) => {
    if (!incomingFiles) return;
    setGlobalError(null);

    const validNewFiles: UploadingFile[] = [];
    for (let i = 0; i < incomingFiles.length; i++) {
      const file = incomingFiles[i];
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (!['jpg', 'jpeg', 'png'].includes(ext || '')) {
        setGlobalError('Only JPG and PNG image files are supported.');
        continue;
      }
      if (file.size > 25 * 1024 * 1024) {
        setGlobalError('File size exceeds the 25MB limit.');
        continue;
      }
      validNewFiles.push({
        file,
        status: 'pending',
        progress: 0,
      });
    }

    setFiles((prev) => [...prev, ...validNewFiles]);
    if (!batchName && validNewFiles.length > 0) {
      const dateStr = new Date().toISOString().replace('T', ' ').substring(0, 16);
      setBatchName(`Scanner Intake ${dateStr}`);
    }
  };

  const removeFile = (index: number) => {
    if (isSubmitting) return;
    setFiles((prev) => prev.filter((_, idx) => idx !== index));
  };

  const startUploadPipeline = async () => {
    if (!batchName.trim()) {
      setGlobalError('Please enter a batch name.');
      return;
    }
    if (files.length === 0) {
      setGlobalError('Please select at least one image to upload.');
      return;
    }

    setIsSubmitting(true);
    setGlobalError(null);

    try {
      // 1. Register batch with backend and get presigned PUT URLs
      const filePayload = files.map((f) => ({
        filename: f.file.name,
        content_type: f.file.type || 'image/jpeg',
        file_size: f.file.size,
      }));

      const batchData = await createBatch(batchName.trim(), filePayload);
      const batchId = batchData.batch_id;

      // Map presigned upload targets to our file state
      const updatedFiles: UploadingFile[] = files.map((f, idx) => ({
        ...f,
        id: batchData.uploads[idx]?.image_id,
        presigned: batchData.uploads[idx],
        status: 'uploading',
        progress: 0,
      }));
      setFiles(updatedFiles);

      // 2. Upload files in parallel directly to S3
      let failureCount = 0;
      let lastErrorMessage = '';
      const uploadPromises = updatedFiles.map(async (f, idx) => {
        if (!f.presigned) return;
        try {
          await uploadFileToS3(
            f.file,
            f.presigned.upload_url,
            f.presigned.content_type,
            (percent) => {
              setFiles((prev) => {
                const next = [...prev];
                if (next[idx]) {
                  next[idx].progress = percent;
                }
                return next;
              });
            }
          );
          setFiles((prev) => {
            const next = [...prev];
            if (next[idx]) {
              next[idx].status = 'completed';
              next[idx].progress = 100;
            }
            return next;
          });
        } catch (err: unknown) {
          failureCount++;
          lastErrorMessage = err instanceof Error ? err.message : 'Upload failed';
          setFiles((prev) => {
            const next = [...prev];
            if (next[idx]) {
              next[idx].status = 'failed';
              next[idx].error = lastErrorMessage;
            }
            return next;
          });
        }
      });

      await Promise.all(uploadPromises);

      if (failureCount === updatedFiles.length) {
        setGlobalError(`All uploads failed: ${lastErrorMessage}`);
        return;
      }

      // 3. Complete batch verification on backend
      await completeBatchUploads(batchId);

      onBatchCreated(batchId);
      onClose();
    } catch (err: unknown) {
      setGlobalError(err instanceof Error ? err.message : 'Upload process failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl shadow-xl border border-slate-200 max-w-2xl w-full p-6 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900">Create New Upload Batch</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Direct S3 presigned upload with strict verification. Zero credentials in browser.
            </p>
          </div>
          {!isSubmitting && (
            <button
              onClick={onClose}
              className="p-1 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {globalError && (
          <div className="mt-4 p-3 rounded-lg bg-rose-50 border border-rose-200 flex items-start gap-2 text-xs text-rose-700">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{globalError}</span>
          </div>
        )}

        <div className="mt-4 space-y-4 overflow-y-auto flex-1 pr-1">
          {/* Batch Name Input */}
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Batch Name / Intake Reference
            </label>
            <input
              type="text"
              value={batchName}
              onChange={(e) => setBatchName(e.target.value)}
              disabled={isSubmitting}
              placeholder="e.g. Scanning Station A - 10:00 AM"
              className="w-full text-sm px-3 py-2 rounded-lg border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
          </div>

          {/* Drag & Drop Area */}
          <div
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFilesAdded(e.dataTransfer.files);
            }}
            className="border-2 border-dashed border-slate-300 hover:border-indigo-500 rounded-xl p-6 text-center cursor-pointer transition bg-slate-50 hover:bg-indigo-50/20"
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".jpg,.jpeg,.png"
              className="hidden"
              onChange={(e) => handleFilesAdded(e.target.files)}
            />
            <Upload className="w-8 h-8 text-indigo-600 mx-auto mb-2" />
            <div className="text-xs font-semibold text-slate-700">
              Click or drag parcel photos here
            </div>
            <div className="text-[11px] text-slate-500 mt-1">
              JPG or PNG formats, up to 25MB each
            </div>
          </div>

          {/* Selected Files List */}
          {files.length > 0 && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-slate-700">
                Selected Files ({files.length})
              </div>
              <div className="divide-y divide-slate-100 max-h-48 overflow-y-auto rounded-lg border border-slate-200">
                {files.map((f, idx) => (
                  <div key={idx} className="p-2.5 flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2 overflow-hidden">
                      <FileText className="w-4 h-4 text-slate-400 shrink-0" />
                      <div className="truncate font-medium text-slate-800">
                        {f.file.name}
                      </div>
                      <span className="text-[10px] text-slate-400 shrink-0">
                        ({(f.file.size / 1024).toFixed(0)} KB)
                      </span>
                    </div>

                    <div className="flex items-center gap-2 shrink-0">
                      {f.status === 'uploading' && (
                        <div className="flex items-center gap-2">
                          <span className="text-[11px] text-indigo-600 font-semibold">
                            {f.progress}%
                          </span>
                          <div className="w-16 bg-slate-200 rounded-full h-1.5">
                            <div
                              className="bg-indigo-600 h-1.5 rounded-full transition-all"
                              style={{ width: `${f.progress}%` }}
                            />
                          </div>
                        </div>
                      )}
                      {f.status === 'completed' && (
                        <span className="inline-flex items-center gap-1 text-[11px] text-emerald-600 font-medium">
                          <Check className="w-3.5 h-3.5" /> Uploaded
                        </span>
                      )}
                      {f.status === 'failed' && (
                        <span className="inline-flex items-center gap-1 text-[11px] text-rose-600 font-medium">
                          <AlertCircle className="w-3.5 h-3.5" /> Failed
                        </span>
                      )}
                      {f.status === 'pending' && !isSubmitting && (
                        <button
                          onClick={() => removeFile(idx)}
                          className="text-slate-400 hover:text-slate-600"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-6 border-t border-slate-100 pt-4 flex items-center justify-end gap-3">
          <button
            onClick={onClose}
            disabled={isSubmitting}
            className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-lg transition"
          >
            Cancel
          </button>
          <button
            onClick={startUploadPipeline}
            disabled={isSubmitting || files.length === 0}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 rounded-lg transition shadow-sm"
          >
            {isSubmitting && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
            {isSubmitting ? 'Uploading to S3...' : `Start Upload (${files.length})`}
          </button>
        </div>
      </div>
    </div>
  );
};
