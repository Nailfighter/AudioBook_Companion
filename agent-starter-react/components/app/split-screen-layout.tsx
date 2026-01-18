'use client';

import { AudioPlayer } from '@/components/app/audio-player';
import { LiveAudioChat } from '@/components/app/live-audio-chat';

export function SplitScreenLayout() {
  return (
    <div className="flex h-screen w-full flex-col md:flex-row">
      {/* Left Half - Audio Player */}
      <div className="flex h-1/2 w-full items-center justify-center md:h-full md:w-1/2">
        <AudioPlayer />
      </div>

      {/* Right Half - Live Audio Chat */}
      <div className="h-1/2 w-full md:h-full md:w-1/2">
        <LiveAudioChat />
      </div>
    </div>
  );
}
