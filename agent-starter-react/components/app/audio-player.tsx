'use client';

import { useEffect, useRef, useState } from 'react';
import { Pause, Play, SkipBack, SkipForward } from '@phosphor-icons/react';

interface Audiobook {
  id: string;
  title: string;
  author: string;
  cover_image: string;
  audio_file: string;
  duration: number;
}

export function AudioPlayer() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);
  const [audiobook, setAudiobook] = useState<Audiobook | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Fetch audiobook data
  useEffect(() => {
    fetch('/audiobooks.json')
      .then((res) => res.json())
      .then((data: Audiobook[]) => {
        if (data.length > 0) {
          const book = data[0];
          setAudiobook(book);
          // Set duration from metadata
          if (book.duration) {
            setDuration(book.duration);
          }
        }
      })
      .catch((error) => console.error('Error loading audiobook data:', error));
  }, []);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const updateDuration = () => {
      if (audio.duration && !isNaN(audio.duration)) {
        setDuration(audio.duration);
      }
    };

    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('loadedmetadata', updateDuration);
    audio.addEventListener('durationchange', updateDuration);
    audio.addEventListener('ended', handleEnded);

    // Set initial duration if already loaded
    if (audio.duration && !isNaN(audio.duration)) {
      setDuration(audio.duration);
    }

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('loadedmetadata', updateDuration);
      audio.removeEventListener('durationchange', updateDuration);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audiobook]);

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const skipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 30);
    }
  };

  const skipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(duration, audioRef.current.currentTime + 30);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime); // Update UI immediately for responsiveness
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
    }
  };

  const changePlaybackSpeed = () => {
    const speeds = [1.0, 1.25, 1.5, 1.75, 2.0];
    const currentIndex = speeds.indexOf(playbackSpeed);
    const nextSpeed = speeds[(currentIndex + 1) % speeds.length];
    setPlaybackSpeed(nextSpeed);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextSpeed;
    }
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return '0:00';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const formatTimeLeft = (seconds: number) => {
    if (isNaN(seconds)) return '0h 0m left';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m left`;
  };

  if (!audiobook) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-white">
        <div className="text-gray-600">Loading audiobook...</div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-white px-4 py-8">
      <div className="w-full max-w-md">
        {/* Book Cover */}
        <div className="mb-6 flex justify-center px-8">
          <div className="aspect-square w-full max-w-sm overflow-hidden rounded-2xl shadow-xl">
            <img
              src={audiobook.cover_image}
              alt={audiobook.title}
              className="h-full w-full object-cover"
            />
          </div>
        </div>

        {/* Book Title and Author */}
        <h2 className="mb-1 px-4 text-center text-lg font-bold text-gray-900">
          {audiobook.title}
        </h2>
        <p className="mb-4 px-4 text-center text-sm text-gray-600">
          by {audiobook.author}
        </p>

        {/* Progress Bar */}
        <div className="mb-1 px-4">
          <div className="relative flex h-1 w-full items-center rounded-full bg-gray-200">
            <div
              className="absolute left-0 top-0 h-full rounded-full bg-orange-400 transition-all"
              style={{ width: `${(currentTime / duration) * 100}%` }}
            ></div>
            {/* Thumb/Circle */}
            <div
              className="absolute h-3 w-3 -translate-x-1/2 rounded-full bg-orange-400 shadow-md transition-all"
              style={{ left: `${(currentTime / duration) * 100}%` }}
            ></div>
            <input
              type="range"
              min="0"
              max={duration || 100}
              value={currentTime}
              onChange={handleSeek}
              className="absolute left-0 top-0 h-full w-full cursor-pointer opacity-0"
            />
          </div>
        </div>

        {/* Time Display */}
        <div className="mb-6 flex justify-between px-4 text-sm text-gray-600">
          <span>{formatTime(currentTime)}</span>
          <span className="font-medium">{formatTimeLeft(duration - currentTime)}</span>
          <span>−{formatTime(duration)}</span>
        </div>

        {/* Player Controls */}
        <div className="mb-6 flex items-center justify-center gap-3 px-4">
          {/* Previous Track */}
          <button
            onClick={skipBackward}
            className="flex h-12 w-12 items-center justify-center rounded-full transition hover:bg-gray-100"
          >
            <SkipBack size={28} weight="fill" />
          </button>

          {/* Rewind 30s */}
          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 30);
              }
            }}
            className="relative flex h-14 w-14 items-center justify-center rounded-full border-2 border-gray-900 transition hover:bg-gray-50"
          >
            <svg
              width="36"
              height="36"
              viewBox="0 0 24 24"
              fill="none"
              stroke="black"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
            <span className="absolute mt-1 text-[8px] font-bold text-black">30</span>
          </button>

          {/* Play/Pause */}
          <button
            onClick={togglePlayPause}
            className="flex h-16 w-16 items-center justify-center rounded-full bg-black text-white transition hover:bg-gray-800"
          >
            {isPlaying ? (
              <Pause size={32} weight="fill" />
            ) : (
              <Play size={32} weight="fill" className="ml-0.5" />
            )}
          </button>

          {/* Forward 30s */}
          <button
            onClick={() => {
              if (audioRef.current) {
                audioRef.current.currentTime = Math.min(
                  duration,
                  audioRef.current.currentTime + 30
                );
              }
            }}
            className="relative flex h-14 w-14 items-center justify-center rounded-full border-2 border-gray-900 transition hover:bg-gray-50"
          >
            <svg
              width="36"
              height="36"
              viewBox="0 0 24 24"
              fill="none"
              stroke="black"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
            </svg>
            <span className="absolute mt-1 text-[8px] font-bold text-black">30</span>
          </button>

          {/* Next Track */}
          <button
            onClick={skipForward}
            className="flex h-12 w-12 items-center justify-center rounded-full transition hover:bg-gray-100"
          >
            <SkipForward size={28} weight="fill" />
          </button>
        </div>

        {/* Bottom Controls */}
        <div className="flex items-start justify-center px-8 pt-4">
          <button
            onClick={changePlaybackSpeed}
            className="flex flex-col items-center gap-1 rounded-lg p-2 transition hover:bg-gray-100"
          >
            <span className="text-lg font-bold text-gray-900">{playbackSpeed}x</span>
            <span className="text-[11px] text-gray-600">Narration speed</span>
          </button>
        </div>

        {/* Hidden Audio Element */}
        <audio ref={audioRef} src={audiobook.audio_file} />
      </div>
    </div>
  );
}
