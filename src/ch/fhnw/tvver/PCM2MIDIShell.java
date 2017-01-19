package ch.fhnw.tvver;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiChannel;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Receiver;
import javax.sound.midi.ShortMessage;
import javax.sound.sampled.AudioFileFormat.Type;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioFormat.Encoding;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.audio.JavaSoundTarget;
import ch.fhnw.ether.audio.NullAudioTarget;
import ch.fhnw.ether.audio.URLAudioSource;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;
import ch.fhnw.ether.midi.MidiToString;
import ch.fhnw.ether.midi.URLMidiSource;
import ch.fhnw.ether.platform.Platform;
import ch.fhnw.ether.ui.ParameterWindow;
import ch.fhnw.ether.ui.ParameterWindow.Flag;
import ch.fhnw.ether.ui.PlotWindow;
import ch.fhnw.tvver.AbstractPCM2MIDI.Flags;
import ch.fhnw.util.ByteList;
import ch.fhnw.util.Log;
import ch.fhnw.util.TextUtilities;
import ch.fhnw.util.color.RGB;

public final class PCM2MIDIShell {
	private static final Log log = Log.create();

	private final static double         SEC2US      = 1000000;

	private double                      time;
	private int                         numTrueDetectedNotes;
	private int                         numFalseDetectedNotes;
	private int                         numRefNotes;
	private final EnumSet<Flags>        flags;
	private ByteList                    pcmOut = new ByteList();
	private MidiChannel                 playbackChannel;
	private List<MidiEvent>             pendingNoteOffs = new LinkedList<MidiEvent>();
	private final RenderProgram<IAudioRenderTarget> program;
	private       IAudioRenderTarget    audioOut;
	private final File                  track;
	boolean[]                           detected = new boolean[128];
	private Set<File>                   midiDump = new HashSet<>();
	final MidiKeyTracker                tracker  = new MidiKeyTracker() {
		@Override
		protected void setVelocity(int key, int velocity) {
			super.setVelocity(key, velocity);
			if(velocity == 0) detected[key] = false;
		};
	};
	private TreeSet<MidiEvent>          midiRef        = new TreeSet<MidiEvent>(URLMidiSource.MIDI_EVWNT_CMP);

	public PCM2MIDIShell(File track, EnumSet<Flags> flags) throws MalformedURLException, IOException, InvalidMidiDataException {		
		this.track = track;
		this.flags = flags;

		URLAudioSource src = new URLAudioSource(track.toURI().toURL(), 1) {
			long lasttime;
			@Override
			protected void sendMidiMsg(Receiver recv, MidiMessage msg, long time) {
				if(time - lastTime() < 0)
					midiDump.add(track);
				if(flags.contains(Flags.DUMP_MIDI) && !(midiDump.contains(track)))
					System.out.println(MidiToString.toString(msg));
				if(msg instanceof ShortMessage && ((ShortMessage)msg).getCommand() == ShortMessage.NOTE_ON) {
					ShortMessage sm = (ShortMessage)msg;
					if(sm.getData2() == 0)
						super.sendMidiMsg(recv, msg, time);
					else if(time != lastTime() && (time - lastTime() >= 1000000 || time - lastTime() < 0)) {
						try {
							lastTime(time);
							msg = new ShortMessage(sm.getStatus(), sm.getData1(), 64);
							super.sendMidiMsg(recv, msg, time);
						} catch (Throwable t) {}
					}
				} else
					super.sendMidiMsg(recv, msg, time);
			}

			private long lastTime()          {return lasttime - Long.MAX_VALUE / 2;}
			private void lastTime(long time) {lasttime = time + Long.MAX_VALUE / 2;}			
		};
		src.getMidiEvents(midiRef);
		for(MidiEvent e : midiRef) {
			MidiMessage m = e.getMessage();
			if(m instanceof ShortMessage && (m.getMessage()[0] & 0xFF) == ShortMessage.NOTE_ON && (m.getMessage()[2] > 0))
				numRefNotes++;
		}

		src.getMidiEvents(midiRef);
		tracker.setRefMidi(midiRef);
		program = new RenderProgram<>(src, tracker);
	}

	public void start(List<AbstractPCM2MIDI> impls, int idx) throws RenderCommandException {
		AbstractPCM2MIDI impl = impls.get(idx);

		impl.initializePipeline(program);

		new ParameterWindow(program, Flag.EXIT_ON_CLOSE, Flag.HIDE_ON_STOP) {
			@Override
			protected void stopped() {
				impl.shutdown();
				super.stopped();
				Platform.get().runOnMainThread(()->{
					if(idx <= impls.size()) {
						try {
							impls.get(idx+1).getShell().start(impls, idx+1);
						} catch(Throwable t) {
							Platform.get().exit();			
						}
					} else {
						Platform.get().exit();			
					}
				});
			}
		};
		if(!(getFlag(Flags.MAX_SPEED)))
			new PlotWindow(program);

		audioOut = impl.getFlag(Flags.MAX_SPEED) ? new NullAudioTarget(1, 44100) : new JavaSoundTarget(pcmOut);
		audioOut.useProgram(program);
		audioOut.start();
	}

	void noteOn(int key, int velocity) {
		time              = audioOut.getFrame().playOutTime;
		double noteOnTime = time;

		try {
			MidiEvent noteOff = new MidiEvent(new ShortMessage(ShortMessage.NOTE_OFF, 0, key), (long) ((noteOnTime + 0.3) * SEC2US));

			if(playbackChannel != null) {
				playbackChannel.noteOn(key, velocity);
				pendingNoteOffs.add(noteOff);
			}

			int[] vs = tracker.getVelocities();

			if(velocity > 0) {
				if(vs[key] > 0) {
					if(!(detected[key])) {
						tracker.bar(key/128f, RGB.GREEN);
						numTrueDetectedNotes++;
						detected[key] = true;
					} else {
						tracker.bar(key/128f, RGB.YELLOW);
					}
				} else {
					tracker.bar(key/128f, RGB.RED);
					numFalseDetectedNotes++;
				}
			}
		} catch(Throwable t) {
			log.warning(t);
		}
	}

	private static final String SEP = "\t";
	String getReport() {
		String result = "";

		result += (int)time + SEP;
		result += numRefNotes + SEP;
		result += + numTrueDetectedNotes + SEP;
		result += + numFalseDetectedNotes + SEP;

		double trueDetectedRatio  = numTrueDetectedNotes /  (double)numRefNotes; 
		double falseDetectedRatio = numFalseDetectedNotes / (double)numRefNotes; 

		result += (1 + (5 * Math.max(0, Math.min(trueDetectedRatio-0.5*falseDetectedRatio, 1)))) + SEP;

		return result;
	}

	private static final String COLUMNS = 
			"File" + SEP + 
			"Track length" + SEP +
			"# of reference notes" + SEP +
			"# of true detected notes" + SEP +
			"# of false detected notes" + SEP +
			"Grade";

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Throwable {
		if(args.length == 0) {
			log.info("Usage: " + PCM2MIDIShell.class.getName() + " <audio_file> <class>");
			System.exit(0);
		}

		Platform.get().init();

		File        src    = new File(args[0]);
		PrintWriter report = new PrintWriter(new File(src.isDirectory() ? src : src.getParentFile(), args[1] + "_report.txt"));

		report.println(COLUMNS);

		Class<AbstractPCM2MIDI> cls = (Class<AbstractPCM2MIDI>)Class.forName("ch.fhnw.tvver." + args[1]);
		List<AbstractPCM2MIDI> pcm2midis = getInputs(src, cls);
		if(pcm2midis.isEmpty()) {
			System.out.println("No inputs found in '"+src+"'");
			System.exit(1);
		}


		pcm2midis.get(0).getShell().start(pcm2midis, 0);

		Platform.get().addShutdownTask(new Runnable() {
			@Override
			public void run() {
				System.out.println("----------" + cls.getName());
				System.out.println(COLUMNS);
				for(AbstractPCM2MIDI pcm2midi : pcm2midis) {
					File src = pcm2midi.getShell().getTrack();
					String row = src.getName() + SEP;
					try {
						if(pcm2midi.getFlag(Flags.REPORT)) {
							row += pcm2midi.getReport();
							report.println(row);
						} if(pcm2midi.getFlag(Flags.WAVE))
							pcm2midi.writeWAV(new File(src.getParent(), TextUtilities.getFileNameWithoutExtension(src) + ".wav"));
					} catch(Throwable t) {
						if(t.getCause() != null) t = t.getCause();
						row += t.getClass().getName() + ":" + t.getMessage() + SEP;
						report.println(row);
					}
					System.out.println(row);
				}
				report.close();
			}
		});

		Platform.get().run();
	}

	protected File getTrack() {
		return track;
	}

	private static List<AbstractPCM2MIDI> getInputs(File fileOrDir, Class<AbstractPCM2MIDI> cls) {
		List<AbstractPCM2MIDI> result = new ArrayList<>();
		getInputs(fileOrDir, cls, result);
		return result;
	}

	private static void getInputs(File fileOrDir, Class<AbstractPCM2MIDI> cls, List<AbstractPCM2MIDI> result) {
		if(fileOrDir.isDirectory()) {
			for(File f : fileOrDir.listFiles())
				getInputs(f, cls, result);
		} else if(fileOrDir.isFile()) {
			try {
				result.add(cls.getConstructor(File.class).newInstance(fileOrDir));
			} catch(Throwable t) {}
		}
	}

	boolean getFlag(Flags flag) {
		return flags.contains(flag);
	}

	void writeWAV(File file) throws IOException, ClassNotFoundException, MidiUnavailableException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		AudioSystem.write(
				new AudioInputStream(new ByteArrayInputStream(pcmOut._getArray(), 0, pcmOut.size()), 
						new AudioFormat(Encoding.PCM_SIGNED, audioOut.getSampleRate(), 16, audioOut.getNumChannels(), audioOut.getNumChannels(), audioOut.getSampleRate(), false), 
						pcmOut.size() / 4), 
				Type.WAVE, 
				file);
	}

	public SortedSet<MidiEvent> getRefMidi() {
		return midiRef;
	}
}
