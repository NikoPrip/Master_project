NameOfData "Grader/Grader"
TypeOfData "MachineSpecification"

!--- Geometry frames

{ "Body"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/MainFrame"
		RGB 0.438 0.260 0
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "FrontAxle"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/FrontAxle"
		RGB 0.043 0.041 0.043
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "AdjustBladeHydraulics"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/AdjustBladeHydraulics"
		RGB 0.154 0.154 0.154
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "Booth"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Booth"
		RGB 0.043 0.041 0.043
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "BoothCap"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/BoothCap"
		RGB 0.438 0.260 0
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "Glass"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Glass"
		RGB 0.427 0.646 0.786
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
		Opacity 0.5
}

{ "Engine"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Engine"
		RGB 0.438 0.260 0
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "EngineParts"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/EngineParts"
		RGB 0.043 0.041 0.043
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "Panel"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Panel"
		RGB 0.034 0.034 0.034
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "Interior"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Interior"
		RGB 0.073 0.064 0.067
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "Frame"
	Position (0, 0, 0)
	RPY (-90, -90, 0)
	ReferenceFrame "Link1"
	Attributes
		GeoID "Geometry/BladeFrameRot"
		RGB 0.438 0.260 0
		Diffuse 0.6
		Specular 1.0
		SpecularPower 20.0
}

{ "BladeFrame"
	Position ( 0, 0, 0)
	RPY (180, 0, 90)
	ReferenceFrame "Link2"
	Attributes
		GeoID "Geometry/BladeFrame"
		RGB 0.438 0.260 0
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
}

{ "ToolGrader"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "TOOL_FRAME"
	Attributes
		GeoID "Geometry/GraderBlade"
		RGB 0.195 0.195 0.195
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
		ScaleWith "Tool Width" 2		
}

{ "ToolMount"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "TOOL_FRAME"
	Attributes
		GeoID "Geometry/BladeMount"
		RGB 0.438 0.260 0
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
		ScaleWith "Tool Width" 2		
}

{ "ToolLeverMount"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "TOOL_FRAME"
	Attributes
		GeoID "Geometry/BladeLeverMount"
		#GeoScale 0.001
		RGB 0.043 0.041 0.043
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
		ScaleWith "Tool Width" 2
}

{ "WearInsert"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "TP"
	Attributes
		GeoID "Geometry/GraderBladeWear"
		RGB 0.195 0.195 0.195
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
		BladeWearPart
}

{ "WheelDiscs"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Discs"
		RGB 0.438 0.260 0.000
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
}

{ "Wheels"
	Position (0, 0, 0)
	RPY (0, 0, 0)
	ReferenceFrame "BodyEndKin"
	Attributes
		GeoID "Geometry/Tires"
		RGB 0.014 0.014 0.014
		Diffuse 0.8
		Specular 0.3
		SpecularPower 100.0
}
