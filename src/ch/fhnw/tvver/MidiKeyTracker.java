package ch.fhnw.tvver;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

import javax.sound.midi.MidiEvent;
import javax.sound.midi.ShortMessage;

import ch.fhnw.ether.audio.IAudioRenderTarget;
import ch.fhnw.ether.media.AbstractRenderCommand;
import ch.fhnw.ether.media.IScheduler;
import ch.fhnw.ether.media.RenderCommandException;
import ch.fhnw.ether.ui.IPlotable;
import ch.fhnw.util.color.RGB;

public class MidiKeyTracker extends AbstractRenderCommand<IAudioRenderTarget> implements IPlotable {
	private final int[] velocities = new int[128];
	private final List<List<MidiEvent>> midiRef = new ArrayList<>(1024*104);
	private       int                   msTime;
	private       SortedSet<MidiEvent>  refMidi = new TreeSet<>();
	
	@Override
	protected void init(IAudioRenderTarget target) throws RenderCommandException {
		super.init(target);
		midiRef.clear();
		for(MidiEvent e : refMidi) {
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
	}
	
	@Override
	protected void run(IAudioRenderTarget target) throws RenderCommandException {
		try {
			int msTimeLimit = (int) (target.getFrame().playOutTime * IScheduler.SEC2MS);
			for(;msTime <= msTimeLimit; msTime++) {
				if(msTime < midiRef.size()) {
					List<MidiEvent> evts = midiRef.get(msTime);
					if(evts != null) {
						for(MidiEvent e : evts) {
							if(e.getMessage() instanceof ShortMessage) {
								ShortMessage sm = (ShortMessage)e.getMessage();
								switch(sm.getCommand()) {
								case ShortMessage.NOTE_ON:
									setVelocity(sm.getData1(), sm.getData2());
									break;
								case ShortMessage.NOTE_OFF:
									setVelocity(sm.getData1(), 0);
									break;
								}
							}
						}
					}
				}
			}
			clear();
			column(velocities, 0, 128, RGB.WHITE);
		} catch(Throwable t) {
			throw new RenderCommandException(t);
		}
	}
	
	protected void setVelocity(int key, int velocity) {
		velocities[key] = velocity;
	}
	
	public int[] getVelocities() {
		return velocities;
	}

	public void setRefMidi(SortedSet<MidiEvent> refMidi) {
		this.refMidi = refMidi;
	}
	
	@Override
	public int getPlotHeight() {
		return 128;
	}
}
