'use client';

import React, { useEffect, useState } from 'react';
import {
  DownloadCloud,
  Sliders,
  Scan,
  FileSpreadsheet,
  QrCode,
  Scale,
  CheckCircle2,
  Clock,
  Sparkles,
  Layers,
  Image as ImageIcon,
} from 'lucide-react';
import { ActiveProcessingItem, BatchResponse } from '@/types';

interface LiveProcessingCardProps {
  batch: BatchResponse;
  activeImage?: ActiveProcessingItem | null;
  totalImages: number;
  processedCount: number;
  queuedCount: number;
  isStreamConnected?: boolean;
}

interface StepDefinition {
  id: string;
  key: string;
  name: string;
  shortName: string;
  icon: React.ElementType;
  description: string;
}

const PIPELINE_STEPS: StepDefinition[] = [
  {
    id: 'step-1',
    key: 'INGESTING',
    name: 'Storage Ingest',
    shortName: 'Ingest',
    icon: DownloadCloud,
    description: 'Downloading scan from S3 private storage',
  },
  {
    id: 'step-2',
    key: 'PREPROCESSING',
    name: 'Blur & Quality Check',
    shortName: 'Preprocess',
    icon: Sliders,
    description: 'Checking sharpness, resolution and orientation',
  },
  {
    id: 'step-3',
    key: 'SCENE_DETECTION',
    name: 'Scene Validation (Branch C)',
    shortName: 'Scene Detect',
    icon: Scan,
    description: 'Zero-shot parcel detection with OWLv2',
  },
  {
    id: 'step-4',
    key: 'METADATA_EXTRACTION',
    name: 'Machine Overlay (Branch A)',
    shortName: 'Overlay OCR',
    icon: FileSpreadsheet,
    description: 'Full-image OCR & layout profile parsing',
  },
  {
    id: 'step-5',
    key: 'LABEL_EXTRACTION',
    name: 'Physical Label & Barcodes (Branch B)',
    shortName: 'Label & Barcode',
    icon: QrCode,
    description: 'Label cropping, deskewing & ZXing barcode decoding',
  },
  {
    id: 'step-6',
    key: 'RECONCILIATION',
    name: 'Evidence Reconciliation',
    shortName: 'Reconciliation',
    icon: Scale,
    description: 'Multi-source arbitration & field confidence scoring',
  },
];

export const LiveProcessingCard: React.FC<LiveProcessingCardProps> = ({
  batch,
  activeImage,
  totalImages,
  processedCount,
  queuedCount,
  isStreamConnected = true,
}) => {
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  // Live timer for elapsed processing seconds on current image
  useEffect(() => {
    if (!activeImage) {
      setElapsedSeconds(0);
      return;
    }

    const startTime = activeImage.stage_started_at
      ? new Date(activeImage.stage_started_at).getTime()
      : Date.now();

    const updateTimer = () => {
      const now = Date.now();
      const diff = Math.max(0, Math.floor((now - startTime) / 1000));
      setElapsedSeconds(diff);
    };

    updateTimer();
    const interval = setInterval(updateTimer, 500);
    return () => clearInterval(interval);
  }, [activeImage?.image_id, activeImage?.stage_started_at]);

  // Determine current active step index (1 to 6)
  const currentStepIndex = React.useMemo(() => {
    if (!activeImage) return 0;
    const stageKey = activeImage.stage;
    const foundIndex = PIPELINE_STEPS.findIndex((s) => s.key === stageKey);
    if (foundIndex !== -1) return foundIndex + 1;
    if (activeImage.stage_index) return activeImage.stage_index;
    return 1;
  }, [activeImage]);

  const currentStep = PIPELINE_STEPS[Math.max(0, currentStepIndex - 1)];
  const currentImageNumber = Math.min(totalImages, processedCount + (activeImage ? 1 : 0));

  return (
    <div className="bg-gradient-to-br from-indigo-950 via-slate-900 to-slate-950 text-white rounded-2xl p-5 md:p-6 shadow-xl border border-indigo-900/50 relative overflow-hidden transition-all duration-300">
      {/* Background ambient glow effect */}
      <div className="absolute top-0 right-0 -mt-16 -mr-16 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 left-1/4 -mb-16 w-72 h-72 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />

      {/* Header bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-indigo-900/60 pb-4 relative z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
            </span>
            <span className="text-xs font-bold uppercase tracking-wider text-emerald-400">
              Live Pipeline Active
            </span>
          </div>
          <span className="text-slate-600">•</span>
          <span className="text-xs font-semibold text-slate-300">
            Image {currentImageNumber} of {totalImages}
          </span>
          {queuedCount > 0 && (
            <span className="inline-flex items-center gap-1 text-[11px] font-mono bg-indigo-900/60 text-indigo-300 border border-indigo-700/50 px-2 py-0.5 rounded-full">
              <Clock className="w-3 h-3 text-indigo-400" />
              {queuedCount} in queue
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-[11px] text-slate-400">
          <span className="inline-flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700/60 font-mono">
            <span className={`w-1.5 h-1.5 rounded-full ${isStreamConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
            {isStreamConnected ? 'Real-time SSE Stream' : 'Live Polling Mode'}
          </span>
          {elapsedSeconds > 0 && (
            <span className="font-mono text-slate-300 bg-slate-800/80 px-2.5 py-1 rounded-lg border border-slate-700/60">
              {elapsedSeconds}s elapsed
            </span>
          )}
        </div>
      </div>

      {/* Main active processing showcase */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 mt-5 relative z-10 items-center">
        {/* Active image preview & details */}
        <div className="lg:col-span-5 flex items-start gap-4 bg-slate-900/80 p-4 rounded-xl border border-indigo-800/40">
          {activeImage?.preview_url ? (
            <div className="relative w-20 h-20 sm:w-24 sm:h-24 rounded-lg overflow-hidden bg-slate-950 border border-indigo-500/30 shrink-0 shadow-inner group">
              <img
                src={activeImage.preview_url}
                alt={activeImage.filename}
                className="w-full h-full object-cover object-center group-hover:scale-105 transition-transform duration-300"
              />
              <div className="absolute inset-0 bg-indigo-900/10 pointer-events-none" />
              <div className="absolute bottom-1 right-1 bg-black/70 text-[9px] font-mono px-1 py-0.2 rounded text-slate-300">
                Live
              </div>
            </div>
          ) : (
            <div className="w-20 h-20 sm:w-24 sm:h-24 rounded-lg bg-indigo-950/70 border border-indigo-800/60 shrink-0 flex items-center justify-center text-indigo-400">
              <ImageIcon className="w-8 h-8 animate-pulse" />
            </div>
          )}

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-1.5 text-[11px] font-mono text-indigo-300">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              <span>CURRENTLY PROCESSING</span>
            </div>
            <h3 className="text-base font-bold text-white truncate mt-0.5" title={activeImage?.filename}>
              {activeImage?.filename || 'Waiting for next image...'}
            </h3>
            <div className="text-[11px] font-mono text-slate-400 truncate mt-0.5">
              ID: {activeImage?.image_id ? activeImage.image_id.substring(0, 8) : '—'}
            </div>

            <div className="mt-2.5 flex items-center gap-2">
              <span className="text-xs font-semibold text-emerald-400 bg-emerald-950/80 border border-emerald-800/50 px-2 py-0.5 rounded">
                Stage {currentStepIndex} of 6
              </span>
              <span className="text-xs text-slate-300 truncate font-medium">
                {currentStep?.name || activeImage?.stage_name || 'Processing'}
              </span>
            </div>
          </div>
        </div>

        {/* Stage description & progress gauge */}
        <div className="lg:col-span-7 flex flex-col justify-between space-y-3 bg-slate-900/50 p-4 rounded-xl border border-indigo-900/30">
          <div>
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-300">Current Pipeline Activity</span>
              <span className="font-mono font-bold text-indigo-400">
                {activeImage?.stage_progress !== undefined ? `${activeImage.stage_progress}%` : `${Math.round((currentStepIndex / 6) * 100)}%`}
              </span>
            </div>
            <p className="text-xs text-indigo-200 mt-1 font-sans">
              {activeImage?.stage_message || currentStep?.description || 'Executing pipeline branches...'}
            </p>
          </div>

          {/* Smooth progress bar */}
          <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden p-0.5 border border-slate-700/60">
            <div
              className="bg-gradient-to-r from-indigo-500 via-blue-500 to-emerald-400 h-1.5 rounded-full transition-all duration-500 ease-out shadow-sm"
              style={{
                width: `${activeImage?.stage_progress !== undefined ? Math.max(5, activeImage.stage_progress) : (currentStepIndex / 6) * 100}%`,
              }}
            />
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
            <span>
              Overall Batch: <strong className="text-slate-200">{processedCount}</strong> / {totalImages} done
            </span>
            <span className="text-slate-400">
              Local CPU M1 • Concurrency 1
            </span>
          </div>
        </div>
      </div>

      {/* Visual step pipeline tracker */}
      <div className="mt-6 pt-5 border-t border-indigo-900/60 relative z-10">
        <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-3">
          Live Stage Progress Tracker
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
          {PIPELINE_STEPS.map((step, idx) => {
            const stepNumber = idx + 1;
            const isCompleted = currentStepIndex > stepNumber;
            const isActive = currentStepIndex === stepNumber;
            const isPending = currentStepIndex < stepNumber;
            const StepIcon = step.icon;

            return (
              <div
                key={step.id}
                className={`relative flex flex-col p-2.5 rounded-xl border transition-all duration-300 ${
                  isActive
                    ? 'bg-indigo-900/70 border-indigo-400 text-white shadow-lg shadow-indigo-950/50 ring-1 ring-indigo-400/50 scale-[1.02]'
                    : isCompleted
                    ? 'bg-slate-900/80 border-emerald-800/60 text-slate-200'
                    : 'bg-slate-900/40 border-slate-800 text-slate-500 opacity-60'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div
                    className={`w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold ${
                      isActive
                        ? 'bg-indigo-600 text-white animate-pulse'
                        : isCompleted
                        ? 'bg-emerald-900/80 text-emerald-300'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                    ) : (
                      <span>{stepNumber}</span>
                    )}
                  </div>

                  <StepIcon
                    className={`w-4 h-4 ${
                      isActive
                        ? 'text-indigo-300 animate-bounce'
                        : isCompleted
                        ? 'text-emerald-400'
                        : 'text-slate-600'
                    }`}
                  />
                </div>

                <div className="text-[11px] font-bold leading-tight truncate">
                  {step.shortName}
                </div>
                <div className="text-[9px] text-slate-400 truncate mt-0.5">
                  {isActive ? (
                    <span className="text-emerald-400 font-semibold">Active now</span>
                  ) : isCompleted ? (
                    <span className="text-emerald-500">Done</span>
                  ) : (
                    <span>Upcoming</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
