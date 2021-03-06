package ch.fhnw.tvver;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import javax.sound.midi.InvalidMidiDataException;
import javax.sound.midi.MidiEvent;
import javax.sound.midi.MidiMessage;
import javax.sound.midi.MidiUnavailableException;
import javax.sound.midi.ShortMessage;
import javax.sound.sampled.UnsupportedAudioFileException;
import javax.swing.JFrame;
import javax.swing.JLabel;

import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.audio.fx.AutoGain;
import ch.fhnw.ether.media.AbstractRenderCommand;
import ch.fhnw.ether.media.IScheduler;
import ch.fhnw.ether.media.Parameter;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.media.RenderProgram;

/**
 * A fake PCM2MIDI implementation which jitters the reference notes
 * and signals the jittered reference notes.
 *
 * @author simon.schubiger@fhnw.ch
 *
 */
public class DataGen extends AbstractPCM2MIDI {
	public DataGen(File track) throws UnsupportedAudioFileException, IOException, MidiUnavailableException, InvalidMidiDataException, RenderCommandException {
		super(track, EnumSet.of(Flags.REPORT, Flags.WAVE));
	}

	@Override
	protected void initializePipeline(RenderProgram<IAudioRenderTarget> program) {
		program.addLast(new AutoGain());
		program.addLast(new PCM2MIDI());

		JFrame frame = new JFrame();
		frame.setVisible(true);
		frame.add(new JLabel("Hello World"));
	}

	public class PCM2MIDI extends AbstractRenderCommand<IAudioRenderTarget> {
		private final List<List<MidiEvent>> midiRef = new ArrayList<>();
		private final List<List<MidiEvent>> midiRefOff = new ArrayList<>();
		private       int                   msTime;

		private HashSet<Integer> currentMidiRef = new HashSet<>();
		private HashSet<Integer> currentMidiRefOff = new HashSet<>();
		private int[] activeMidi = new int[128];

		private List<Float> currentSamples = new LinkedList<>();

		private int lastWriteTime = 0;
		private BufferedWriter buffer;
		private int hertz = 44100;
		private int msOffset = 50;
		private int msLength = 100;

		public PCM2MIDI() {
			super();
		}

		@Override
		protected void init(IAudioRenderTarget target) throws RenderCommandException {
			super.init(target);
			midiRef.clear();
			for(MidiEvent e : getRefMidi()) {
				MidiMessage msg = e.getMessage();
				if(msg instanceof ShortMessage &&
						(msg.getMessage()[0] & 0xFF) != ShortMessage.NOTE_ON ||
						(msg.getMessage()[2] & 0xFF) == 0) continue;
				int msTime = (int) (e.getTick() / 1000L);
				while(midiRef.size() <= msTime)
					midiRef.add(null);
				List<MidiEvent> evts = midiRef.get(msTime);
				if(evts == null) {
					evts = new ArrayList<MidiEvent>();
					midiRef.set(msTime, evts);
				}
				evts.add(e);
			}

			for(MidiEvent e : getRefMidi()) {
				MidiMessage msg = e.getMessage();
				if(msg instanceof ShortMessage &&
						(msg.getMessage()[0] & 0xFF) != ShortMessage.NOTE_OFF &&
						((msg.getMessage()[0] & 0xFF) != ShortMessage.NOTE_ON || (msg.getMessage()[2] & 0xFF) != 0)) continue;
				int msTime = (int) (e.getTick() / 1000L);
				while(midiRefOff.size() <= msTime)
					midiRefOff.add(null);
				List<MidiEvent> evts = midiRefOff.get(msTime);
				if(evts == null) {
					evts = new ArrayList<MidiEvent>();
					midiRefOff.set(msTime, evts);
				}
				evts.add(e);
			}
			File file = new File("audiotest");
			try {
				buffer = Files.newBufferedWriter(Paths.get(file.getAbsolutePath()));
			} catch (IOException e1) {
			}
		}

		@Override
		protected void run(IAudioRenderTarget target) throws RenderCommandException {
			try {
				int msTimeLimit = (int) (target.getFrame().playOutTime * IScheduler.SEC2MS);
				
				for(float f : target.getFrame().getMonoSamples()){ 
					currentSamples.add(f);
				}
				
				for(;msTime <= msTimeLimit; msTime++) {
					if(msTime < midiRef.size()) {
						List<MidiEvent> evts = midiRef.get(msTime);
						if(evts != null) {
							for(MidiEvent e : evts){
								currentMidiRef.add((int) e.getMessage().getMessage()[1]);  //Liste der aktuell aktiven Midi-Werte
							}
						}
					}
					if(msTime < midiRefOff.size()) {
						List<MidiEvent> evts = midiRefOff.get(msTime);
						if(evts != null) {
							for(MidiEvent e : evts){
								currentMidiRefOff.add((int) e.getMessage().getMessage()[1]);  //Liste der aktuell aktiven Midi-Werte
							}
						}
					}

					//buffer.write("audiotest start");
					//buffer.newLine();
					if(msTime - lastWriteTime >= msOffset && msTime > msLength){
						lastWriteTime = Math.max(lastWriteTime + msOffset, msLength);
						
						List<Float> curSamples = new ArrayList<Float>(currentSamples.subList(0, hertz * msLength / 1000));
						currentSamples = new ArrayList<Float>(currentSamples.subList(hertz * msOffset / 1000, currentSamples.size()));

						int [] midiRefArray = new int[128];

						for(Integer mr : currentMidiRef){
							if(activeMidi[mr] < 2){
								midiRefArray[mr] = 1;
								activeMidi[mr]++;
							}
						}

						buffer.write(curSamples.stream()
								.map(f -> String.format("%f", f))
								.collect(Collectors.joining(",")) 
								+" " 
								+ Arrays.toString(midiRefArray).replace(" ", "").replace("[", "").replace("]", ""));
						buffer.newLine();
						//File.Write(String.join(",", currentSamples) + "---" + String.join(",", currentMidiRef) + "\n");  //Pseudocode!
						//currentSamples = new LinkedList<Float>();
						//currentMidiRef = new HashSet<Integer>();
						for(Integer r : currentMidiRefOff){
							currentMidiRef.remove(r);
							activeMidi[r] = 0;
						}

						currentMidiRefOff = new HashSet<>();


						buffer.flush();
						//buffer.close();
					}
				}
			} catch(Throwable t) {
				throw new RenderCommandException(t);
			}
		}
	}

}
