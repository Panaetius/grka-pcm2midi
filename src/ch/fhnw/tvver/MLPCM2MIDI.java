package ch.fhnw.tvver;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

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
import org.apache.xmlrpc.*;
import org.apache.xmlrpc.client.*;

/**
 * A fake PCM2MIDI implementation which jitters the reference notes
 * and signals the jittered reference notes.
 *
 * @author simon.schubiger@fhnw.ch
 *
 */
public class MLPCM2MIDI extends AbstractPCM2MIDI {
	public MLPCM2MIDI(File track) throws UnsupportedAudioFileException, IOException, MidiUnavailableException, InvalidMidiDataException, RenderCommandException {
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
		private       int                   msTime;

		private List<Float> currentSamples = new LinkedList<>();
		private XmlRpcClient server = new XmlRpcClient();

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
			XmlRpcClientConfigImpl config = new XmlRpcClientConfigImpl();
		    try {
				config.setServerURL(new URL("http://localhost:8765/RPC2"));
//				config.setEnabledForExtensions(true);
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		    server.setConfig(config);
		}

		@Override
		protected void run(IAudioRenderTarget target) throws RenderCommandException {
			try {
				int msTimeLimit = (int) (target.getFrame().playOutTime * IScheduler.SEC2MS);
				
				for(float f : target.getFrame().getMonoSamples()){ 
					currentSamples.add(f);
				}
				
				for(;msTime <= msTimeLimit; msTime++) {
					

					if(msTime - lastWriteTime >= msOffset && msTime > msLength){
						lastWriteTime = Math.max(lastWriteTime + msOffset, msLength);
						
						List<Float> curSamples = new ArrayList<Float>(currentSamples.subList(0, hertz * msLength / 1000));
						currentSamples = new ArrayList<Float>(currentSamples.subList(hertz * msOffset / 1000, currentSamples.size()));
						
//						HashMap<String,List<Float>> params = new HashMap<String,List<Float>>();
//						params.put("images", curSamples);
//						Object[] requestParams = new Object[] { null };
//						requestParams[0] = params;
						List<Double> doubleList = new ArrayList<Double>();
						
						for(float f : curSamples){
							doubleList.add((double)f);
						}
						Object[] params = new Object[]{doubleList};
						
						Integer result = (Integer)server.execute("inference", params);
						
						if(result < 128){
							noteOn(result, 100);
						}
						
					}
				}
			} catch(Throwable t) {
				throw new RenderCommandException(t);
			}
		}
	}

}
