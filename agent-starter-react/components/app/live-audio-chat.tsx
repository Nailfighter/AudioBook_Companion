'use client';

import { useEffect, useRef, useState } from 'react';
import {
  useSessionContext,
  useSessionMessages,
  useVoiceAssistant,
} from '@livekit/components-react';
import { Microphone, MicrophoneSlash, Phone, PhoneDisconnect } from '@phosphor-icons/react';
import { cn } from '@/lib/utils';
import { ScrollArea } from '@/components/livekit/scroll-area/scroll-area';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'agent';
  timestamp: Date;
}

export function LiveAudioChat() {
  const session = useSessionContext();
  const { messages: livekitMessages } = useSessionMessages(session);
  const { state: agentState } = useVoiceAssistant();
  const [messages, setMessages] = useState<Message[]>([]);
  const [isMicMuted, setIsMicMuted] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Convert LiveKit messages to our message format
  useEffect(() => {
    const convertedMessages: Message[] = livekitMessages.map((msg, index) => ({
      id: `${msg.timestamp}-${index}`,
      text: msg.message || '',
      sender: msg.from?.isLocal ? 'user' : 'agent',
      timestamp: new Date(msg.timestamp),
    }));
    setMessages(convertedMessages);

    // Auto-scroll to bottom
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [livekitMessages]);

  const handleConnect = () => {
    if (session.isConnected) {
      session.end();
    } else {
      session.start();
    }
  };

  const handleMicToggle = async () => {
    if (session.room?.localParticipant) {
      const audioTrack = session.room.localParticipant.getTrackPublication(
        'microphone'
      );
      if (audioTrack) {
        await session.room.localParticipant.setMicrophoneEnabled(!isMicMuted);
        setIsMicMuted(!isMicMuted);
      }
    }
  };

  const getAgentStatusText = () => {
    switch (agentState) {
      case 'listening':
        return 'Agent is listening...';
      case 'thinking':
        return 'Agent is thinking...';
      case 'speaking':
        return 'Agent is speaking...';
      default:
        return 'Ready to chat';
    }
  };

  const getAgentStatusColor = () => {
    switch (agentState) {
      case 'listening':
        return 'bg-green-500';
      case 'thinking':
        return 'bg-yellow-500';
      case 'speaking':
        return 'bg-blue-500';
      default:
        return 'bg-gray-400';
    }
  };

  return (
    <div className="flex h-full flex-col bg-white">
      {/* Header */}
      <div className="border-b border-gray-200 px-6 py-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-900">AI Assistant</h2>
            <div className="mt-1 flex items-center gap-2">
              <div className={cn('h-2 w-2 rounded-full', getAgentStatusColor())} />
              <p className="text-sm text-gray-600">{getAgentStatusText()}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Connection Status Badge */}
            <div
              className={cn(
                'rounded-full px-3 py-1 text-xs font-medium',
                session.isConnected
                  ? 'bg-gray-900 text-white'
                  : 'bg-gray-200 text-gray-700'
              )}
            >
              {session.isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <ScrollArea ref={scrollAreaRef} className="flex-1 px-6 py-4">
        {!session.isConnected && messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center text-center">
            <div className="mb-6 rounded-full bg-gray-100 p-8">
              <Phone size={56} className="text-gray-900" weight="fill" />
            </div>
            <h3 className="mb-2 text-lg font-bold text-gray-900">
              Connect to AI Assistant
            </h3>
            <p className="mb-6 max-w-sm text-sm text-gray-600">
              Start a conversation with your AI companion. Ask questions, get summaries, or
              discuss the audiobook you're listening to.
            </p>
          </div>
        )}

        {messages.length > 0 && (
          <div className="space-y-3">
            {messages.map((message) => (
              <div
                key={message.id}
                className={cn(
                  'flex',
                  message.sender === 'user' ? 'justify-end' : 'justify-start'
                )}
              >
                <div
                  className={cn(
                    'max-w-[80%] rounded-2xl px-4 py-3',
                    message.sender === 'user'
                      ? 'bg-gray-900 text-white'
                      : 'bg-gray-100 text-gray-900'
                  )}
                >
                  <p className="text-sm leading-relaxed">{message.text}</p>
                  <p
                    className={cn(
                      'mt-1 text-xs',
                      message.sender === 'user' ? 'text-gray-400' : 'text-gray-500'
                    )}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Agent Typing Indicator */}
        {agentState === 'thinking' && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-2xl bg-gray-100 px-4 py-3">
              <div className="flex gap-1">
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-600" />
                <div
                  className="h-2 w-2 animate-bounce rounded-full bg-gray-600"
                  style={{ animationDelay: '0.1s' }}
                />
                <div
                  className="h-2 w-2 animate-bounce rounded-full bg-gray-600"
                  style={{ animationDelay: '0.2s' }}
                />
              </div>
            </div>
          </div>
        )}
      </ScrollArea>

      {/* Audio Visualizer (when speaking or listening) */}
      {session.isConnected && (agentState === 'speaking' || agentState === 'listening') && (
        <div className="border-t border-gray-200 bg-gray-50 px-6 py-4">
          <div className="flex items-center justify-center gap-1">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="bg-gray-900 w-1 rounded-full"
                style={{
                  height: `${Math.random() * 40 + 10}px`,
                  animationName: 'pulse',
                  animationDuration: `${0.5 + Math.random() * 0.5}s`,
                  animationTimingFunction: 'ease-in-out',
                  animationIterationCount: 'infinite',
                  animationDelay: `${i * 0.05}s`,
                }}
              />
            ))}
          </div>
        </div>
      )}

      {/* Control Bar */}
      <div className="border-t border-gray-200 bg-white px-6 py-6">
        <div className="flex items-center justify-center gap-3">
          {/* Microphone Toggle */}
          <button
            onClick={handleMicToggle}
            disabled={!session.isConnected}
            className={cn(
              'flex h-14 w-14 items-center justify-center rounded-full border-2 transition-all',
              session.isConnected
                ? isMicMuted
                  ? 'border-gray-900 bg-gray-900 text-white hover:bg-gray-800'
                  : 'border-gray-900 bg-white text-gray-900 hover:bg-gray-50'
                : 'border-gray-300 bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
          >
            {isMicMuted ? (
              <MicrophoneSlash size={24} weight="fill" />
            ) : (
              <Microphone size={24} weight="fill" />
            )}
          </button>

          {/* Connect/Disconnect Button */}
          <button
            onClick={handleConnect}
            className={cn(
              'flex h-16 w-16 items-center justify-center rounded-full transition-all',
              session.isConnected
                ? 'bg-gray-900 text-white hover:bg-gray-800'
                : 'bg-gray-900 text-white hover:bg-gray-800'
            )}
          >
            {session.isConnected ? (
              <PhoneDisconnect size={28} weight="fill" />
            ) : (
              <Phone size={28} weight="fill" />
            )}
          </button>
        </div>

        {/* Helper Text */}
        <p className="mt-4 text-center text-xs text-gray-600">
          {session.isConnected
            ? 'Speak naturally - the AI can hear you'
            : 'Click the phone button to start chatting'}
        </p>
      </div>
    </div>
  );
}
