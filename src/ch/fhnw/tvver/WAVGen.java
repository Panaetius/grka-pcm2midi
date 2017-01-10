package ch.fhnw.tvver;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.InvocationTargetException;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiChannel;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiSystem;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.Sequence;
import javax.sound.midi.Sequencer;
import javax.sound.midi.Synthesizer;
import javax.sound.midi.Track;
import javax.sound.midi.spi.MidiFileReader;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.Clip;
import javax.sound.sampled.LineUnavailableException;
import javax.sound.sampled.UnsupportedAudioFileException;
import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioFileFormat.Type;
import javax.sound.sampled.AudioFormat.Encoding;
import javax.sound.sampled.spi.FormatConversionProvider;

import ch.fhnw.ether.audio.FileAudioTarget;
import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.audio.JavaSoundTarget;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;
import ch.fhnw.tvver.AbstractPCM2MIDI.Flags;
import ch.fhnw.util.ByteList;
import ch.fhnw.util.Log;

public class WAVGen extends AbstractPCM2MIDI {
	
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
	private final static EnumSet<Flags> flags = EnumSet.of(Flags.WAVE, Flags.DEBUG);;
	private ByteList                    pcmOut = new ByteList();
	private MidiChannel                 playbackChannel;
	private Sequence                    midiSeq;
	private Track                       midiTrack;
	private List<MidiEvent>             pendingNoteOffs = new LinkedList<MidiEvent>();
	//private final RenderProgram<IAudioRenderTarget> program;
	private       JavaSoundTarget       audioOut;
	final MidiKeyTracker                tracker        = new MidiKeyTracker();
	private TreeSet<MidiEvent>          midiRef        = new TreeSet<MidiEvent>(new Comparator<MidiEvent>() {
		@Override
		public int compare(MidiEvent o1, MidiEvent o2) {
			int   result  = (int) (o1.getTick() - o2.getTick());
			return result == 0 ? o1.getMessage().getMessage()[1] - o2.getMessage().getMessage()[1] : result;
		}
	});
	
	protected WAVGen(File track, EnumSet<Flags> flags) throws UnsupportedAudioFileException, IOException,
			MidiUnavailableException, InvalidMidiDataException, RenderCommandException {
		super(track, flags);
		// TODO Auto-generated constructor stub
	}

	

	@Override
	protected void initializePipeline(RenderProgram<IAudioRenderTarget> program) {
		// TODO Auto-generated method stub
		
	}
	
	protected void generateWAV(File file) throws IOException, ClassNotFoundException, MidiUnavailableException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException {
			audioOut = new JavaSoundTarget();
			AudioSystem.write(
					new AudioInputStream(new ByteArrayInputStream(pcmOut._getArray(), 0, pcmOut.size()), 
							audioOut.getJavaSoundAudioFormat(), 
							pcmOut.size() / 4), 
					Type.WAVE, 
					file);
	}
	
	public static void main (String[] args) throws UnsupportedAudioFileException, IOException, MidiUnavailableException, InvalidMidiDataException, RenderCommandException, ClassNotFoundException, IllegalAccessException, IllegalArgumentException, InvocationTargetException, NoSuchMethodException, SecurityException, LineUnavailableException {
				
		//File file = new File("scale.mid");
		//WAVGen wavGen = new WAVGen(file, flags);
		//SortedSet<MidiEvent> midiEvents = wavGen.getRefMidi();		
		
		
		InputStream targetStream = new FileInputStream("scale.mid");
		
		//MidiReader midireader = new MidiReader();
		//midireader.getSequence(targetStream);
		
		Sequencer seq1 = MidiSystem.getSequencer();
		seq1.setSequence(targetStream);
		seq1.open();
		/*seq1.start();
		seq1.startRecording();
		*/
		
		
		Sequence sequence = seq1.getSequence();
		Track track = sequence.createTrack();
		
		Synthesizer synth1 = MidiSystem.getSynthesizer();
		synth1.open();
		MidiChannel[] midichannels = synth1.getChannels();
		midichannels[0].noteOn(88, 100);
		//synth1.close();
		
		
		
		
		Clip clip  = AudioSystem.getClip();
		//clip.open();
		System.out.println(clip.getFormat());
		Byte[] bytearray = new Byte[1024];
		System.out.println(clip.getBufferSize());
		bytearray = turnClipIntoByteArray(clip);
		
		File audiotest = new File("audiotest");
		OutputStream clipStream = new FileOutputStream(audiotest);
		////clipStream.write(bytearray);
		
		//AudioInputStream audiostream = ...;
		
		//AudioSystem.write(audioStream, AudioFileFormat.Type.WAVE, audiotest);
		FileAudioTarget filetarget = new FileAudioTarget(audiotest, 16, 441000);
		//filetarget.render();
		//filetarget.stop();
		
		
		
	}
	
	static Byte[] turnClipIntoByteArray(Clip clip){
		Byte[] bytearray;
		System.out.println(clip.available());
		
		return null;
	}
	
}
