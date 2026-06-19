from opentrons import protocol_api
from opentrons.types import Point

# --- 1. METADATA ---
metadata = {
    'apiLevel': '2.14',
    'protocolName': 'Autonomous Fluid Drop (Offset Edition)',
    'description': 'Dispenses a specific volume of liquid into a custom offset coordinate in Slot 1.'
}

def run(protocol: protocol_api.ProtocolContext):
    
    # --- 2. CONFIGURATION VARIABLES ---
    DISPENSE_VOLUME = 100  # uL
    OFFSET_X = 15.0 
    OFFSET_Y = 60.0
    HOVER_HEIGHT = 65.0 # Height above the top lip of the slot
    

    target_rig = protocol.load_labware('nest_1_reservoir_195ml', '1') #slot 1 resevoir
    
    fluid_source = protocol.load_labware('nest_1_reservoir_195ml', '7') # where its getting fluid from
    
    tiprack = protocol.load_labware('opentrons_96_tiprack_1000ul', '11') #tiprack
    
    # --- 4. LOAD PIPETTE ---
    pipette = protocol.load_instrument('p1000_single_gen2', 'left', tip_racks=[tiprack])
    
    # --- 5. THE DISPENSE SEQUENCE ---
    protocol.comment(f"dropping: {DISPENSE_VOLUME}uL to (X:{OFFSET_X}, Y:{OFFSET_Y}) in Slot 1")
    
    pipette.pick_up_tip()
    
    # Aspirate from the bottom of the water source (hovering 2mm above the plastic bottom)
    pipette.aspirate(DISPENSE_VOLUME, fluid_source.wells()[0].bottom(z=2))
    
    custom_drop_location = target_rig.wells()[0].top(z=HOVER_HEIGHT).move(Point(x=OFFSET_X, y=OFFSET_Y, z=0.0))
    
    # Move to the custom rig and dispense
    pipette.dispense(DISPENSE_VOLUME, custom_drop_location)
    
    # Return tip to the rack to save consumables
    pipette.return_tip()
    
    protocol.comment("Done")