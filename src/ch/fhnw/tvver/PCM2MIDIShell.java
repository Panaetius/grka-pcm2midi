package ch.fhnw.tvver;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.net.MalformedURLException;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.LinkedList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiChannel;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Sequence;
import javax.sound.midi.ShortMessage;
import javax.sound.midi.Track;
import javax.sound.sampled.AudioFileFormat.Type;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;

import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.audio.JavaSoundTarget;
import ch.fhnw.ether.audio.URLAudioSource;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;
import ch.fhnw.ether.platform.Platform;
import ch.fhnw.ether.ui.ParameterWindow;
import ch.fhnw.ether.ui.PlotWindow;
import ch.fhnw.ether.ui.ParameterWindow.Flag;
import ch.fhnw.tvver.AbstractPCM2MIDI.Flags;
import ch.fhnw.util.ByteList;
import ch.fhnw.util.IntList;
import ch.fhnw.util.Log;
import ch.fhnw.util.TextUtilities;

public final class PCM2MIDIShell {
	private static final Log log = Log.create();

	private final static double         SEC2US      = 1000000;
	private final static double         US2SEC      = 1 / SEC2US;
	private final static double         MAX_LATENCY = 0.1;

	private double                      time;
	private int                         numDetectedNotes;
	private int                         numTrueDetectedNotes;
	private int                         numFalseDetectedNotes;
	private int                         numRefNotes;
	private double                      minLat = MAX_LATENCY;
	private double                      maxLat;
	private double                      sumLat;
	private final EnumSet<Flags>        flags;
	private ByteList                    pcmOut = new ByteList();
	private MidiChannel                 playbackChannel;
	private Sequence                    midiSeq;
	private Track                       midiTrack;
	private List<MidiEvent>             pendingNoteOffs = new LinkedList<MidiEvent>();
	private final RenderProgram<IAudioRenderTarget> program;
	private       JavaSoundTarget       audioOut;
	final MidiKeyTracker                tracker        = new MidiKeyTracker();
	private TreeSet<MidiEvent>          midiRef        = new TreeSet<MidiEvent>(new Comparator<MidiEvent>() {
		@Override
		public int compare(MidiEvent o1, MidiEvent o2) {
			int   result  = (int) (o1.getTick() - o2.getTick());
			return result == 0 ? o1.getMessage().getMessage()[1] - o2.getMessage().getMessage()[1] : result;
		}
	});

	public PCM2MIDIShell(File track, EnumSet<Flags> flags) throws MalformedURLException, IOException, InvalidMidiDataException {
		this.flags = flags;

		if(flags.contains(Flags.WAVE) || flags.contains(Flags.REPORT)) {
			midiSeq   = new Sequence(Sequence.SMPTE_25, (int) (SEC2US / Sequence.SMPTE_25));
			midiTrack = midiSeq.createTrack(); 
		}

		URLAudioSource src = new URLAudioSource(track.toURI().toURL(), 1);
		src.getMidiEvents(midiRef);
		for(MidiEvent e : midiRef) {
			MidiMessage m = e.getMessage();
			if(m instanceof ShortMessage && (m.getMessage()[0] & 0xFF) == ShortMessage.NOTE_ON && (m.getMessage()[2] > 0))
				numRefNotes++;
		}

		tracker.setRefMidi(midiRef);
		program = new RenderProgram<>(src, tracker);
	}

	public void start(AbstractPCM2MIDI impl) throws RenderCommandException {
		impl.initializePipeline(program);

		new ParameterWindow(program, Flag.EXIT_ON_CLOSE, Flag.CLOSE_ON_STOP);
		new PlotWindow(program);
		
		audioOut = new JavaSoundTarget(pcmOut);
		audioOut.useProgram(program);
		audioOut.start();
	}

	void noteOn(int key, int velocity) {
		time              = audioOut.getFrame().playOutTime;
		double noteOnTime = time;

		try {
			MidiEvent noteOn  = new MidiEvent(new ShortMessage(ShortMessage.NOTE_ON,  0, key, velocity), (long) (noteOnTime * SEC2US));
			MidiEvent noteOff = new MidiEvent(new ShortMessage(ShortMessage.NOTE_OFF, 0, key), (long) ((noteOnTime + 0.3) * SEC2US));

			if(playbackChannel != null) {
				playbackChannel.noteOn(key, velocity);
				pendingNoteOffs.add(noteOff);
			}

			if(midiTrack != null) {
				midiTrack.add(noteOn);
				midiTrack.add(noteOff);

				for(MidiEvent e : midiRef.headSet(noteOn, true).descendingSet()) {
					double lat = (noteOn.getTick() - e.getTick()) * US2SEC;
					if(lat > MAX_LATENCY) {
						numFalseDetectedNotes++;
						break;
					}

					if(noteOn.getMessage().getMessage()[1] == e.getMessage().getMessage()[1]) {
						minLat = Math.min(minLat, lat);
						maxLat = Math.max(maxLat, lat);
						sumLat += lat;
						numDetectedNotes++;
						numTrueDetectedNotes++;
						break;
					}
				} 
			}
			IntList notes = new IntList();
			int[]   vs    = tracker.getVelocities();
			for(int i = 0; i < vs.length; i++)
				if(vs[i] > 0) notes.add(i);
			if(flags.contains(Flags.DEBUG))
				System.out.println(key + ":" + notes);
		} catch(Throwable t) {
			log.warning(t);
		}
	}

	private static final String SEP = "\t";
	String getReport() {
		if(maxLat == 0 && numDetectedNotes == 0) maxLat = MAX_LATENCY;

		String result = "";

		double avgLat = numDetectedNotes == 0 ? 0 : sumLat / numDetectedNotes;
		result += time + SEP;
		result += numRefNotes + SEP;
		result += + numTrueDetectedNotes + SEP;
		result += + numFalseDetectedNotes + SEP;
		result += + (int)(minLat * 1000) + SEP;
		result += (int)(maxLat * 1000) + SEP;
		result += (int)(avgLat * 1000) + SEP;

		double trueDetectedRatio  = numTrueDetectedNotes /  (double)numRefNotes; 
		double falseDetectedRatio = numFalseDetectedNotes / (double)numRefNotes; 

		result += (1 + (5 * Math.max(0, Math.min(trueDetectedRatio-falseDetectedRatio, 1)))) + SEP;

		return result;
	}

	private static final String COLUMNS = 
			"File" + SEP + 
			"Track length" + SEP +
			"# of reference notes" + SEP +
			"# of true detected notes" + SEP +
			"# of false detected notes" + SEP +
			"Min. Latency" + SEP +
			"Max. Latency" + SEP +
			"Avg. Latency" + SEP +
			"Grade";

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Throwable {
		if(args.length == 0) {
			log.info("Usage: " + PCM2MIDIShell.class.getName() + " <audio_file> <class>");
			System.exit(0);
		}

		Platform.get().init();

		File        src    = new File(args[0]);
		PrintWriter report = new PrintWriter(new File(src.getParent(), args[1] + "_report.txt"));

		report.println(COLUMNS);

		Class<AbstractPCM2MIDI> cls = (Class<AbstractPCM2MIDI>)Class.forName("ch.fhnw.tvver." + args[1]);

		AbstractPCM2MIDI pcm2midi = cls.getConstructor(File.class).newInstance(src);
		pcm2midi.getShell().start(pcm2midi);

		Platform.get().addShutdownTask(new Runnable() {
			@Override
			public void run() {
				if(src.isFile() && !src.getName().endsWith(".txt") && !src.getName().startsWith(".")) {
					String row = src.getName() + SEP;
					System.out.println("----------" + src.getName());
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
					report.close();
				}
			}
		});		

		Platform.get().run();
	}

	boolean getFlag(Flags flag) {
		return flags.contains(flag);
	}

	void writeWAV(File file) throws IOException, ClassNotFoundException, MidiUnavailableException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
		AudioSystem.write(
				new AudioInputStream(new ByteArrayInputStream(pcmOut._getArray(), 0, pcmOut.size()), 
						audioOut.getJavaSoundAudioFormat(), 
						pcmOut.size() / 4), 
				Type.WAVE, 
				file);
	}

	public SortedSet<MidiEvent> getRefMidi() {
		return midiRef;
	}
}
